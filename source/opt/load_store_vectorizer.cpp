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
#include "cfa.h"

#include <unordered_set>

const uint32_t kStorePtrIdInIdx = 0;
const uint32_t kLoadPtrIdInIdx = 0;
const uint32_t kAccessChainPtrIdInIdx = 0;
const uint32_t kCopyObjectOperandInIdx = 0;
const uint32_t kTypePointerStorageClassInIdx = 0;
const uint32_t kTypePointerTypeIdInIdx = 1;
const uint32_t kOpConstantListeralIndex = 0;

const uint32_t kStorageClassIdx = 1;
const uint32_t kOperandTypeIdx = 2;

// todo:
//  - support loads
//  - add common subexpression elimination to make this pattern work:
//		- array[idx + 1].x/y/z/w
//	- improve & fix dead code stripping
// bugs:
//

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

  std::map<const ir::BasicBlock*, std::unique_ptr<ir::BasicBlock>>
      replaceBlocks;

  auto post_order = [&](const ir::BasicBlock* bi) {
    uint32_t instIdx = 0;
    InstrListMap storeOps;

    for (auto ii = bi->cbegin(); ii != bi->cend(); ++ii, ++instIdx) {
      switch (ii->opcode()) {
        case SpvOpStore: {
          uint32_t varId;
          ir::Instruction* ptrInst = GetPtr((ir::Instruction*)&*ii, &varId);
          storeOps[varId].push_back(&*ii);
        } break;
      }
    }

    std::vector<ir::Instruction> basicBlockInstructions;
    for (auto ii = bi->cbegin(); ii != bi->cend(); ++ii)
      basicBlockInstructions.push_back(*ii);

    bool localChanged = VectorizeChains(&basicBlockInstructions, storeOps);
    globalChanged |= localChanged;

    if (localChanged) {
      std::unique_ptr<ir::BasicBlock> newBB(
          new ir::BasicBlock(std::move(NewLabel(bi->id()))));

      for (auto& i : basicBlockInstructions) {
        newBB->AddInstruction(
            std::unique_ptr<ir::Instruction>(new ir::Instruction(i)));
      }
      newBB->SetParent(func);
      replaceBlocks[&*bi] = std::move(newBB);
      ++bi;
    } else {
      ++bi;
    }
  };

  ComputeStructuredSuccessors(func);

  auto ignore_block = [](const ir::BasicBlock*) {};
  auto ignore_edge = [](const ir::BasicBlock*, const ir::BasicBlock*) {};
  auto get_structured_successors = [this](const ir::BasicBlock* block) {
    return &(block2structured_succs_[block]);
  };

  CFA<ir::BasicBlock>::DepthFirstTraversal(
      &*func->begin(), get_structured_successors, ignore_block, post_order,
      ignore_edge);

  // patch up all basic blocks with their new ops
  for (auto bi = func->begin(); bi != func->end();) {
    auto biPtr = &*bi;
    if (replaceBlocks.count(biPtr)) {
      bi = bi.Erase();
      bi = bi.InsertBefore(std::move(replaceBlocks[biPtr]));
    }
    ++bi;
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
      std::vector<const ir::Instruction*> chunk(
          chain.second.begin() + CI, chain.second.begin() + CI + len);
      changed |= VectorizeInstructions(block_ptr, chunk);
    }
  }

  return changed;
}

bool LoadStoreVectorizerPass::IsConsecutiveAccess(const ir::Instruction* a,
                                                  const ir::Instruction* b) {
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

      bool opsAreSame = opA == opB;
      if (!opsAreSame) {
        // the ops are different, check if they are OpLoad's by chance
        // and if they try to load the same data; this may indicate
        // array indexing so we want to treat them the same too.
        // jb-todo: determine if they are the same sub-expression instead
        // because right now array[idx + 1].x/y/z/w breaks this pattern.
        opsAreSame = AreIdenticalLoads(opA, opB);
      }

      sameStart &= opsAreSame;
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
    InstVec* bbInstrs, std::vector<const ir::Instruction*>& instrs) {
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

  std::set<const ir::Instruction*> instructionsProcessed;
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
    std::vector<const ir::Instruction*> chainOperands;
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
        VectorizeStoreChain(bbInstrs, chainOperands, &instructionsProcessed);

    changed |= vectorized;
  }

  return changed;
}

bool LoadStoreVectorizerPass::VectorizeStoreChain(
    InstVec* bbInstrs, std::vector<const ir::Instruction*> chainOperands,
    std::set<const ir::Instruction*>* processed) {
  // 0. Find or create an OpTypeVector
  // 1. Create an OpCompositeConstruct with all the same arguments as the
  // OpStores that are in the operands list
  // 2. Replace all OpStore and OpAccessChain's leading here, with a single
  // OpStore and OpAccessChain
  processed->insert(chainOperands.begin(), chainOperands.end());

  auto opTypeVector = FindVectorInOpAccessChain(chainOperands[0]);

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

      ir::Instruction newComposite(SpvOpCompositeConstruct, 0,
                                   opTypeVector->result_id(), ops);

      instructions.push_back(newComposite);
    }

    uint32_t opTypePointerId = TakeNextId();
    {
      std::unique_ptr<ir::Instruction> opTypePointer(new ir::Instruction(
          SpvOpTypePointer, 0, opTypePointerId,
          {{spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
            {uint32_t(SpvStorageClassUniform)}},
           {spv_operand_type_t::SPV_OPERAND_TYPE_ID,
            {opTypeVector->result_id()}}}));

      auto foundIt = std::find_if(
          module_->types_values_begin(), module_->types_values_end(),
          [&opTypePointer](const ir::Instruction& a) {
            return a.opcode() == opTypePointer->opcode() &&
                   a.GetSingleWordOperand(kStorageClassIdx) ==
                       opTypePointer->GetSingleWordOperand(kStorageClassIdx) &&
                   a.GetSingleWordOperand(kOperandTypeIdx) ==
                       opTypePointer->GetSingleWordOperand(kOperandTypeIdx);
          });

      if (foundIt == module_->types_values_end())
        module_->AddType(std::move(opTypePointer));
      else
        opTypePointerId = foundIt->result_id();
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
    auto insertPoint =
        FindInBasicBlock(bbInstrs, *chainOperands[chainOperands.size() - 1]);
    bbInstrs->insert(insertPoint, instructions.begin(), instructions.end());

    // just nop out the now redundant stores & access chains
    for (auto& oper : chainOperands) {
      uint32_t varId;
      const ir::Instruction* ptrInst = GetPtr(oper, &varId);

      // nop out store in basic block copy
      auto storeOp = &*FindInBasicBlock(bbInstrs, *oper);
      storeOp->ToNop();

      if (IsNonPtrAccessChain(ptrInst->opcode())) {
        auto ptrOp = &*FindInBasicBlock(bbInstrs, *ptrInst);

        // if we have multiple uses we just leave the access chain in;
        // we don't update def_use_mgr_ right now since i'm not sure if it's
        // correct to do so
        auto uses = def_use_mgr_->GetUses(ptrInst->result_id());
        if (uses->size() == 1 && uses->begin()->inst == &*oper) {
          // nop out OpAccessChain in the basicblock copy
          ptrOp->ToNop();
        }
      }
    }

    return true;
  }

  return false;
}

ir::Instruction* LoadStoreVectorizerPass::FindVectorInOpAccessChain(
    const ir::Instruction* opLoadOrOpStore) {
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
    foundVectorInstruction = nullptr;
    const uint32_t curWord = opAccessChain->GetSingleWordOperand(i);
    // Earlier ID checks ensure that cur_word definition exists.
    auto curWordInstr = def_use_mgr_->GetDef(curWord);
    switch (typePointedTo->opcode()) {
      case SpvOpTypeMatrix:
      case SpvOpTypeVector:
      case SpvOpTypeArray:
      case SpvOpTypeRuntimeArray: {
        // todo: didn't test Array / Matrix yet
        if (typePointedTo->opcode() == SpvOpTypeVector)
          foundVectorInstruction = typePointedTo;
        typePointedTo =
            def_use_mgr_->GetDef(typePointedTo->GetSingleWordOperand(1));
        break;
      }
      case SpvOpTypeStruct: {
        const uint32_t curIndex = curWordInstr->GetSingleWordOperand(2);
        auto structMemberId = typePointedTo->GetSingleWordOperand(curIndex + 1);
        typePointedTo = def_use_mgr_->GetDef(structMemberId);
        break;
      }
      default: { return nullptr; }
    }
  }

  return foundVectorInstruction;
}

// todo: move this to DefUseManager? stolen from MemPass
inline bool LoadStoreVectorizerPass::IsNonPtrAccessChain(
    const SpvOp opcode) const {
  return opcode == SpvOpAccessChain || opcode == SpvOpInBoundsAccessChain;
}

// todo: move this to DefUseManager? stolen from MemPass
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

inline ir::Instruction* LoadStoreVectorizerPass::GetPtr(
    const ir::Instruction* ip, uint32_t* varId) {
  const SpvOp op = ip->opcode();
  assert(op == SpvOpStore || op == SpvOpLoad);
  const uint32_t ptrId = ip->GetSingleWordInOperand(
      op == SpvOpStore ? kStorePtrIdInIdx : kLoadPtrIdInIdx);
  return GetPtr(ptrId, varId);
}

// todo: move this to DefUseManager? stolen from MemPass
std::vector<ir::Instruction>::iterator
LoadStoreVectorizerPass::FindInBasicBlock(InstVec* bbInstrs,
                                          const ir::Instruction& toFind) {
  auto foundIt = std::find_if(
      bbInstrs->begin(), bbInstrs->end(), [toFind](const ir::Instruction& a) {
        bool ok = a.opcode() == toFind.opcode() &&
                  a.result_id() == toFind.result_id() &&
                  a.type_id() == toFind.type_id() &&
                  std::equal(a.begin(), a.end(), toFind.begin(), toFind.end());

        return ok;
      });

  return foundIt;
}

bool LoadStoreVectorizerPass::AreIdenticalLoads(const ir::Operand& opA,
                                                const ir::Operand& opB) {
  ir::Instruction* instrA = def_use_mgr_->GetDef(opA.words[0]);
  ir::Instruction* instrB = def_use_mgr_->GetDef(opB.words[0]);

  if (instrA->opcode() != SpvOpLoad || instrB->opcode() != SpvOpLoad)
    return false;

  // this shouldn't happen due to SSA; but just in case, these are identical
  if (instrA->result_id() == instrB->result_id()) return true;

  uint32_t typeA = instrA->type_id();
  uint32_t typeB = instrB->type_id();

  uint32_t varA = instrA->GetSingleWordInOperand(0);
  uint32_t varB = instrB->GetSingleWordInOperand(0);

  if (typeA == typeB && varA == varB) return true;

  return false;
}

}  // namespace opt
}  // namespace spvtools
