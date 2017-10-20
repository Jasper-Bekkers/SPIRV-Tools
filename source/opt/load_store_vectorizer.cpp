// Copyright (c) 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "load_store_vectorizer.h"

#include <unordered_set>

const uint32_t kStorePtrIdInIdx = 0;
const uint32_t kLoadPtrIdInIdx = 0;
const uint32_t kAccessChainPtrIdInIdx = 0;
const uint32_t kCopyObjectOperandInIdx = 0;
const uint32_t kTypePointerStorageClassInIdx = 0;
const uint32_t kTypePointerTypeIdInIdx = 1;
const uint32_t kOpConstantListeralIndex = 0;

template <typename R, typename E>
bool is_contained(R&& Range, const E& Element) {
  return std::find(Range.begin(), Range.end(), Element) != Range.end();
}

namespace spvtools {
namespace opt {

Pass::Status LoadStoreVectorizerPass::Process(ir::Module* module) {
  InitializeInline(module);

  // Identify live functions first.  Those that are not live
  // are dead.
  ProcessFunction pfn = [this](ir::Function* fp) { return RunOnFunction(fp); };
  bool modified = ProcessEntryPointCallTree(pfn, module_);

  FinalizeNextId(module_);

  return modified ? Pass::Status::SuccessWithChange
                  : Pass::Status::SuccessWithoutChange;
}

bool LoadStoreVectorizerPass::RunOnFunction(ir::Function* func) {
  bool globalChanged = false;

  // llvm does this post_order
  for (auto bi = func->begin(); bi != func->end();) {
    uint32_t instIdx = 0;
    InstrListMap storeOps;

    for (auto ii = bi->begin(); ii != bi->end(); ++ii, ++instIdx) {
      switch (ii->opcode()) {
        case SpvOpStore: {
          uint32_t varId;
          ir::Instruction* ptrInst = GetPtr(&*ii, &varId);
          storeOps[varId].push_back(&*ii);
        } break;
      }
    }

    std::vector<ir::Instruction> basicBlockInstructions(bi->begin(), bi->end());

    bool localChanged = VectorizeChains(&basicBlockInstructions, storeOps);
    globalChanged |= localChanged;

    if (localChanged) {
      std::unique_ptr<ir::BasicBlock> newBB(
          new ir::BasicBlock(std::move(NewLabel(bi->id()))));

      for (auto& i : basicBlockInstructions) {
        newBB->AddInstruction(
            std::unique_ptr<ir::Instruction>(new ir::Instruction(i)));
      }

      bi = bi.Erase();
      bi = bi.InsertBefore(std::move(newBB));
      ++bi;
    } else {
      ++bi;
    }
  }

  return globalChanged;
}

bool LoadStoreVectorizerPass::VectorizeChains(InstVec* block_ptr,
                                              InstrListMap& map) {
  bool changed = false;

  for (auto& chain : map) {
    size_t size = chain.second.size();
    if (size < 2) continue;

    for (unsigned CI = 0, CE = size; CI < CE; CI += 64) {
      unsigned len = std::min<unsigned>(CE - CI, 64);
      std::vector<ir::Instruction*> chunk(chain.second.begin() + CI,
                                          chain.second.begin() + CI + len);
      changed |= VectorizeInstructions(block_ptr, chunk);
    }
  }

  return changed;
}

bool LoadStoreVectorizerPass::IsConsecutiveAccess(ir::Instruction* a,
                                                  ir::Instruction* b) {
  uint32_t varAId, varBId;
  ir::Instruction* ptrA = GetPtr(a, &varAId);
  ir::Instruction* ptrB = GetPtr(b, &varBId);

  if (!ptrA || !ptrB) return false;

  if (varAId != varBId) return false;

  if (ptrA->opcode() == SpvOpAccessChain &&
      ptrB->opcode() == SpvOpAccessChain) {
    // check if
    // - type matches
    // - variable matches
    // - indices match exactly, except for the last one
    // - the last indices are consecutive

    // do this by just checking all the dwords except for the last one (this
    // gets rid of the first 3 checks)

    if (ptrA->NumOperands() != ptrB->NumOperands()) return false;

    uint32_t numOps = ptrA->NumOperands();
    bool sameStart = true;
    for (uint32_t opIdx = 0; opIdx < numOps - 1; opIdx++) {
      const ir::Operand& opA = ptrA->GetOperand(opIdx);
      const ir::Operand& opB = ptrB->GetOperand(opIdx);

      if (opA.type != opA.type) sameStart = false;
      // don't check the ResultId since it will never match
      if (opA.type == SPV_OPERAND_TYPE_RESULT_ID) continue;
      if (opB.type == SPV_OPERAND_TYPE_RESULT_ID) continue;

      sameStart &= opA == opB;
    }

    if (!sameStart) return false;

    const ir::Operand& opAEnd = ptrA->GetOperand(numOps - 1);
    const ir::Operand& opBEnd = ptrB->GetOperand(numOps - 1);
    ir::Instruction* aInst = def_use_mgr_->GetDef(opAEnd.words[0]);
    ir::Instruction* bInst = def_use_mgr_->GetDef(opBEnd.words[0]);

    if (aInst->opcode() == SpvOpConstant && bInst->opcode() == SpvOpConstant) {
      uint32_t aValue = aInst->GetSingleWordInOperand(kOpConstantListeralIndex);
      uint32_t bValue = bInst->GetSingleWordInOperand(kOpConstantListeralIndex);

      // SPIR-V doesn't store byte offsets but element offsets so we can just
      // compare with 1 here
      return ((int)bValue - (int)aValue) == 1;
    }
  }

  return false;
}

bool LoadStoreVectorizerPass::VectorizeInstructions(
    InstVec* bbInstrs, std::vector<ir::Instruction*>& instrs) {
  std::vector<int> heads, tails;
  int consecutiveChain[64];

  // Do a quadratic search on all of the given stores and find all of the pairs
  // of stores that follow each other.
  for (int i = 0, e = instrs.size(); i < e; ++i) {
    consecutiveChain[i] = -1;
    for (int j = e - 1; j >= 0; --j) {
      if (i == j) continue;

      if (IsConsecutiveAccess(instrs[i], instrs[j])) {
        if (consecutiveChain[i] != -1) {
          int curDistance = std::abs(consecutiveChain[i] - i);
          int newDistance = std::abs(consecutiveChain[i] - j);
          if (j < i || newDistance > curDistance)
            continue;  // Should not insert.
        }

        tails.push_back(j);
        heads.push_back(i);
        consecutiveChain[i] = j;
      }
    }
  }

  std::set<ir::Instruction*> instructionsProcessed;
  bool changed = false;
  for (int head : heads) {
    if (instructionsProcessed.count(instrs[head])) continue;
    bool longerChainExists = false;
    for (unsigned TIt = 0; TIt < tails.size(); TIt++)
      if (head == tails[TIt] &&
          !instructionsProcessed.count(instrs[heads[TIt]])) {
        longerChainExists = true;
        break;
      }
    if (longerChainExists) continue;

    // We found an instr that starts a chain. Now follow the chain and try to
    // vectorize it.
    std::vector<ir::Instruction*> chainOperands;
    int I = head;
    while (I != -1 && (is_contained(tails, I) || is_contained(heads, I))) {
      if (instructionsProcessed.count(instrs[I])) break;

      chainOperands.push_back(instrs[I]);
      I = consecutiveChain[I];
    }

    bool vectorized = false;
    // if (isa<LoadInst>(*Operands.begin()))
    //	Vectorized = vectorizeLoadChain(Operands, &InstructionsProcessed);
    // else
    vectorized =
        vectorizeStoreChain(bbInstrs, chainOperands, &instructionsProcessed);

    changed |= vectorized;
  }

  return changed;
}

bool LoadStoreVectorizerPass::vectorizeStoreChain(
    InstVec* bbInstrs, std::vector<ir::Instruction*> chainOperands,
    std::set<ir::Instruction*>* processed) {
  // 0. Find or create an OpTypeVector
  // 1. Create an OpCompositeConstruct with all the same arguments as the
  // OpStores that are in the operands list
  // 2. Replace all OpStore and OpAccessChain's leading here, with a single
  // OpStore and OpAccessChain
  processed->insert(chainOperands.begin(), chainOperands.end());

  auto opTypeVector = findVectorInOpAccessChain(chainOperands[0]);

  if (opTypeVector) {
    std::vector<ir::Instruction> instructions;

    uint32_t opConstantCompositeId = TakeNextId();
    {
      std::vector<ir::Operand> ops{
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {opConstantCompositeId}}};

      // steal the OpStore operands and put them in an OpConstantComposite
      for (auto& k : chainOperands) {
        ops.push_back(k->GetOperand(k->NumOperands() - 1));
      }

      ir::Instruction newComposite(SpvOpConstantComposite, 0,
                                   opTypeVector->result_id(), ops);

      instructions.push_back(newComposite);
    }

    uint32_t opTypePointerId = TakeNextId();
    {
      ir::Instruction type_inst(
          SpvOpTypePointer, 0, opTypePointerId,
          {{spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
            {uint32_t(SpvStorageClassUniform)}},
           {spv_operand_type_t::SPV_OPERAND_TYPE_ID,
            {opTypeVector->result_id()}}});
      instructions.push_back(type_inst);
    }

    // 1. take the old access chain, chop off the last index
    // 2. rebuild it with the new OpTypePointer that we constructed
    uint32_t opAccessChainId = TakeNextId();
    {
      uint32_t dummy;
      ir::Instruction* oldAC = GetPtr(chainOperands[0], &dummy);

      std::vector<ir::Operand> newACOps{
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {opAccessChainId}}};
      newACOps.insert(newACOps.end(), oldAC->begin() + 2, oldAC->end() - 1);

      ir::Instruction newAccessStore(SpvOpAccessChain, 0, opTypePointerId,
                                     newACOps);
      instructions.push_back(newAccessStore);
    }

    // Emit the vectorized store
    {
      ir::Instruction newStore(
          SpvOpStore, 0, 0,
          {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {opAccessChainId}},
           {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {opConstantCompositeId}}});

      instructions.push_back(newStore);
    }

    // insert our new op-code sequence
    auto insertPoint = FindInBasicBlock(bbInstrs, *chainOperands[0]);
    bbInstrs->insert(insertPoint, instructions.begin(), instructions.end());

    // just nop out the now redundant stores & access chains
    for (auto& oper : chainOperands) {
      uint32_t dummy;
      ir::Instruction* opAccessChain = GetPtr(oper, &dummy);
      auto foundIt = FindInBasicBlock(bbInstrs, *opAccessChain);
      *foundIt = ir::Instruction(SpvOpNop, 0, 0, {});

      foundIt = FindInBasicBlock(bbInstrs, *oper);
      *foundIt = ir::Instruction(SpvOpNop, 0, 0, {});
    }

    return true;
  }

  return false;
}

ir::Instruction* LoadStoreVectorizerPass::findVectorInOpAccessChain(
    ir::Instruction* opLoadOrOpStore) {
  uint32_t dummy;
  ir::Instruction* opAccessChain = GetPtr(opLoadOrOpStore, &dummy);

  ir::Instruction* foundVectorInstruction = nullptr;
  // Base must be a pointer, pointing to the base of a composite object.
  auto baseIdIndex = 2;
  auto baseInstr =
      def_use_mgr_->GetDef(opAccessChain->GetSingleWordOperand(baseIdIndex));
  auto baseTypeInstr = def_use_mgr_->GetDef(baseInstr->GetSingleWordOperand(0));

  auto typePointedTo =
      def_use_mgr_->GetDef(baseTypeInstr->GetSingleWordOperand(2));
  for (size_t i = 3; i < opAccessChain->NumOperands(); ++i) {
    const uint32_t curWord = opAccessChain->GetSingleWordOperand(i);
    // Earlier ID checks ensure that cur_word definition exists.
    auto curWordInstr = def_use_mgr_->GetDef(curWord);
    switch (typePointedTo->opcode()) {
      case SpvOpTypeMatrix:
      case SpvOpTypeVector:
      case SpvOpTypeArray:
      case SpvOpTypeRuntimeArray: {
        // In OpTypeMatrix, OpTypeVector, OpTypeArray, and OpTypeRuntimeArray,
        // word 2 is the Element Type.
        if (typePointedTo->opcode() == SpvOpTypeVector) {
          foundVectorInstruction = typePointedTo;
        }

        typePointedTo =
            def_use_mgr_->GetDef(typePointedTo->GetSingleWordOperand(1));
        break;
      }
      case SpvOpTypeStruct: {
        const uint32_t curIndex = curWordInstr->GetSingleWordOperand(2);
        auto structMemberId =
            typePointedTo->GetSingleWordOperand(curIndex + 1);
        typePointedTo = def_use_mgr_->GetDef(structMemberId);
        break;
      }
      default: { return false; }
    }
  }

  return foundVectorInstruction;
}

inline bool LoadStoreVectorizerPass::IsNonPtrAccessChain(
    const SpvOp opcode) const {
  return opcode == SpvOpAccessChain || opcode == SpvOpInBoundsAccessChain;
}

inline ir::Instruction* LoadStoreVectorizerPass::GetPtr(uint32_t ptrId,
                                                        uint32_t* varId) {
  *varId = ptrId;
  ir::Instruction* ptrInst = def_use_mgr_->GetDef(*varId);
  while (ptrInst->opcode() == SpvOpCopyObject) {
    *varId = ptrInst->GetSingleWordInOperand(kCopyObjectOperandInIdx);
    ptrInst = def_use_mgr_->GetDef(*varId);
  }
  ir::Instruction* varInst = ptrInst;
  while (varInst->opcode() != SpvOpVariable &&
         varInst->opcode() != SpvOpFunctionParameter) {
    if (IsNonPtrAccessChain(varInst->opcode())) {
      *varId = varInst->GetSingleWordInOperand(kAccessChainPtrIdInIdx);
    } else {
      assert(varInst->opcode() == SpvOpCopyObject);
      *varId = varInst->GetSingleWordInOperand(kCopyObjectOperandInIdx);
    }
    varInst = def_use_mgr_->GetDef(*varId);
  }
  return ptrInst;
}

inline ir::Instruction* LoadStoreVectorizerPass::GetPtr(ir::Instruction* ip,
                                                        uint32_t* varId) {
  const SpvOp op = ip->opcode();
  assert(op == SpvOpStore || op == SpvOpLoad);
  const uint32_t ptrId = ip->GetSingleWordInOperand(
      op == SpvOpStore ? kStorePtrIdInIdx : kLoadPtrIdInIdx);
  return GetPtr(ptrId, varId);
}

}  // namespace opt
}  // namespace spvtools
