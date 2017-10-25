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

#include "struct_vectorizer.h"
#include "make_unique.h"
#include "opcode.h"

// stretch goal:
//  - support mixed types (eg. f32/f32/int/int where we emit floatToIntBits ops
//  for half of these)
//  - support nested structs

const uint32_t kFloatBitdepthIndex = 1;
const uint32_t kConstantValueIndex = 2;
const uint32_t kIntWidthIndex = 1;
namespace spvtools {
namespace opt {

void StructVectorizerPass::FindAccessChains(
    uint32_t id, std::vector<ir::Instruction*>* outOpChains) {
  auto uses = def_use_mgr_->GetUses(id);
  if (uses) {
    for (auto& ii : *uses) {
      auto instr = ii.inst;
      if (IsNonPtrAccessChain(instr->opcode())) {
        outOpChains->push_back(instr);
      }

      // traverse type hierarchy; potentially keep track of depth
      if (spvOpcodeGeneratesType(instr->opcode()) ||
          spvOpcodeReturnsLogicalPointer(
              instr->opcode())) {  // we probably also need to handle pointer
                                   // variables here
        FindAccessChains(instr->result_id(), outOpChains);
      }
    }
  }
}

bool StructVectorizerPass::GatherStructSpans(ir::Instruction* structOp,
                                             std::vector<Span>* outSpans) {
  // try to generate new spans for adjacent of floats
  // 1. if the alignment of the first float is a multiple of 16
  // 2. the members next to it are tightly packed on 4 byte boundaries
  // 3. all members are of the same type
  // then we can create a new span.
  //
  // this function should return true if we've managed to create at least one
  // new span
  // this function should return a list of all spans that can be vectorized and
  // all spans that can't (so we can re-assemble the struct properly)
  using namespace analysis;

  Type* t = type_mgr_->GetType(structOp->result_id());

  if (Struct* s = t->AsStruct()) {
    auto& elementTypes = s->element_types();

    const uint32_t vectorElementCount = 4;

    uint32_t numSpansToVectorize = 0;
    for (uint32_t baseElementIter = 0; baseElementIter < elementTypes.size();) {
      Float* baseFloat = elementTypes[baseElementIter]->AsFloat();
      uint32_t baseOffset = s->GetElementOffset(baseElementIter);

      Span foundSpan = {elementTypes[baseElementIter], baseElementIter,
                        baseOffset, 1, false};

      if (!baseFloat) {
        outSpans->push_back(foundSpan);
        baseElementIter++;
        continue;
      }

      uint32_t widthInBytes = baseFloat->width() / 8;
      uint32_t naturalAlignment = widthInBytes * vectorElementCount;

      if (baseOffset % naturalAlignment == 0 &&
          widthInBytes == 4 /* untested for other sizes */) {
        for (uint32_t spanElementIter = baseElementIter + 1;
             spanElementIter < elementTypes.size(); spanElementIter++) {
          uint32_t elOffset = s->GetElementOffset(spanElementIter);

          // elements are directly adjacent in memory
          bool continueSpan =
              (elOffset == baseOffset + foundSpan.count * widthInBytes);

          // and are all of the same type as base
          // todo: support mixed types
          continueSpan &= elementTypes[spanElementIter]->IsSame(
              elementTypes[baseElementIter]);

          if (!continueSpan) {
            break;
          }

          foundSpan.count++;

          if (foundSpan.count == vectorElementCount) {
            break;
          }
        }

        if (foundSpan.count == vectorElementCount) {
          foundSpan.shouldVectorize = true;
          numSpansToVectorize++;
        } else {
          // didn't find anything worth while, just make it a span of 1 and
          // continue the search
          foundSpan.count = 1;
        }

        baseElementIter += foundSpan.count;
        outSpans->push_back(foundSpan);
      } else {
        outSpans->push_back(foundSpan);
        baseElementIter++;
      }
    }
    return numSpansToVectorize > 0;
  }

  return false;
}

// transforms struct { float x,y,z,w; } to struct { vec4 data; }
Pass::Status StructVectorizerPass::Process(ir::Module* module) {
  bool modified = false;
  module_ = module;
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module));
  type_mgr_.reset(new analysis::TypeManager(consumer(), *module));
  InitNextId();
  FindNamedOrDecoratedIds();

  std::vector<std::unique_ptr<ir::Instruction>> structs;
  for (auto& instr : module->types_values()) {
    if (instr.opcode() == SpvOpTypeStruct) {
      structs.push_back(MakeUnique<ir::Instruction>(instr));
    }
  }

  for (auto& s : structs) {
    std::vector<Span> spans;
    if (GatherStructSpans(&*s, &spans)) {
      // 1. create a vec4 type if it doesn't exist yet
      // 2. replace the struct content from 4 floats, to the vec4
      // 3. patch up all access chains to point to the vec4, so need to insert
      // an extra index in the chain
      auto floatId = SafeCreateFloatType();

      PatchAccessChains(&*s, spans);

      // type creation:
      // 1. find or create a float 32
      // 2. find or create a vec4 of float32
      // 3. replace struct we found

      auto vectorId = SafeCreateVectorId(floatId, 4 /* hardcode 4 components*/);

      uint32_t structId = GenerateNewStruct(&*s, spans, vectorId);

      KillNamesAndDecorates(&*s);

      auto uses = def_use_mgr_->GetUses(s->result_id());
      if (uses) {
        std::vector<ir::Instruction*> killList;
        for (auto& instr : *uses) {
          switch (instr.inst->opcode()) {
            case SpvOpMemberName:
            case SpvOpMemberDecorate:
              killList.push_back(instr.inst);

              break;
          }
        }
        for (auto& k : killList) def_use_mgr_->KillInst(k);
      }

      def_use_mgr_->ReplaceAllUsesWith(s->result_id(), structId);
      def_use_mgr_->KillInst(&*s);

      MoveTypesDownRecursively(structId);

      modified = true;
    }
  }

  FinalizeNextId();
  return modified ? Pass::Status::SuccessWithChange
                  : Pass::Status::SuccessWithoutChange;
}

uint32_t StructVectorizerPass::GenerateNewStruct(ir::Instruction* s,
                                                 const std::vector<Span>& spans,
                                                 uint32_t vectorId) {
  auto structId = TakeNextId();

  std::vector<ir::Operand> structMembers;

  for (auto& span : spans) {
    if (!span.shouldVectorize) {
      auto op = s->GetOperand(span.typeIdx + 1);
      structMembers.push_back(op);
    } else {
      ir::Operand op = {spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                        {uint32_t(vectorId)}};
      structMembers.push_back(op);
    }

    // todo: properly emit other OpMemberDecorate's that were part of this
    // struct?

    std::unique_ptr<ir::Instruction> opMemberDecorate(new ir::Instruction(
        SpvOpMemberDecorate, structId, 0,
        {
            {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
             {uint32_t(structMembers.size() - 1)}},
            {spv_operand_type_t::SPV_OPERAND_TYPE_DECORATION,
             {uint32_t(SpvDecorationOffset)}},
            {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
             {uint32_t(span.baseOffset)}},
        }));

    module_->AddAnnotationInst(std::move(opMemberDecorate));
  }

  std::unique_ptr<ir::Instruction> opStruct(
      new ir::Instruction(SpvOpTypeStruct, 0, structId, structMembers));

  module_->AddType(std::move(opStruct));

  return structId;
}

uint32_t StructVectorizerPass::MakeUint32() {
  auto type_id = TakeNextId();
  ir::Operand widthOperand(spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                           {32});
  ir::Operand signOperand(spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                          {0});
  std::unique_ptr<ir::Instruction> newType(new ir::Instruction(
      SpvOp::SpvOpTypeInt, type_id, 0, {widthOperand, signOperand}));
  module_->AddType(std::move(newType));
  return type_id;
}

uint32_t StructVectorizerPass::MakeConstantInt(uint32_t value) {
  uint32_t constantId = TakeNextId();

  uint32_t intTypeId = 0;

  {
    auto t = module_->GetTypes();
    auto foundIt = std::find_if(t.begin(), t.end(), [](ir::Instruction* instr) {
      if (instr->opcode() == SpvOpTypeInt) {
        if (instr->GetSingleWordOperand(kIntWidthIndex) == 32) return true;
      }

      return false;
    });

    if (foundIt != t.end()) {
      intTypeId = (*foundIt)->result_id();
    } else {
      intTypeId = MakeUint32();
    }
  }

  std::unique_ptr<ir::Instruction> c(new ir::Instruction(
      SpvOpConstant, intTypeId, constantId,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {value}}}));

  module_->AddGlobalValue(std::move(c));

  return constantId;
}

void StructVectorizerPass::PatchAccessChains(ir::Instruction* s,
                                             const std::vector<Span>& spans) {
  std::vector<ir::Instruction*> accessChains;
  FindAccessChains(s->result_id(), &accessChains);

  std::vector<std::tuple<Span, uint32_t, ir::Instruction*>>
      vectorizeAccessChains;
  std::vector<std::pair<uint32_t, ir::Instruction*>> remapAccessChains;
  for (auto& chain : accessChains) {
    auto last = def_use_mgr_->GetDef(
        chain->GetSingleWordOperand(chain->NumOperands() - 1));

    assert(last->opcode() == SpvOpConstant);

    uint32_t offset = last->GetSingleWordOperand(kConstantValueIndex);

    for (uint32_t remapIdx = 0; remapIdx < spans.size(); remapIdx++) {
      auto& span = spans[remapIdx];
      if (span.shouldVectorize) {
        if (offset >= span.typeIdx && offset < span.typeIdx + span.count) {
          vectorizeAccessChains.push_back(
              std::make_tuple(span, remapIdx, chain));
          break;
        }
      } else {
        // other then vectorizing the access-chains that we turned into vector
        // elements, we also need to re-number all the struct members that came
        // after each new vector member
        if (offset == span.typeIdx && remapIdx != span.typeIdx) {
          remapAccessChains.push_back(std::make_pair(remapIdx, chain));
          break;
        }
      }
    }
  }

  // patch up access chains
  for (auto& kv : vectorizeAccessChains) {
    Span span;
    uint32_t remapIdx;
    ir::Instruction* opAC;
    std::tie(span, remapIdx, opAC) = kv;

    uint32_t indexOffset = MakeConstantInt(remapIdx);

    std::vector<ir::Operand> ops(opAC->begin(), opAC->end());
    ops.insert(ops.end() - 1,
               {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {indexOffset}});

    auto oldC = def_use_mgr_->GetDef(ops[ops.size() - 1].words[0])
                    ->GetSingleWordOperand(kConstantValueIndex);

    auto newC = oldC - span.typeIdx;

    ops[ops.size() - 1].words[0] = MakeConstantInt(newC);

    opAC->ReplaceOperands(ops);
  }

  for (auto& kv : remapAccessChains) {
    auto& remapIdx = kv.first;
    auto& opAC = kv.second;

    uint32_t newLastIndex = MakeConstantInt(remapIdx);

    std::vector<ir::Operand> ops(opAC->begin(), opAC->end());
    ops[ops.size() - 1].words[0] = newLastIndex;

    opAC->ReplaceOperands(ops);
  }
}

void StructVectorizerPass::MoveTypesDownRecursively(uint32_t typeId) {
  auto uses = def_use_mgr_->GetUses(typeId);
  if (uses) {
    for (auto& t : *uses) {
      if (spvOpcodeGeneratesType(t.inst->opcode()) ||
          t.inst->opcode() == SpvOpVariable) {
        std::unique_ptr<ir::Instruction> newInst(new ir::Instruction(*t.inst));
        auto newId = newInst->result_id();
        module_->AddType(std::move(newInst));
        t.inst->ToNop();

        MoveTypesDownRecursively(newId);
      }
    }
  }
}

uint32_t StructVectorizerPass::SafeCreateVectorId(uint32_t floatId,
                                                  uint32_t numComponents) {
  auto vectorId = TakeNextId();

  std::unique_ptr<ir::Instruction> opVec4(new ir::Instruction(
      SpvOpTypeVector, 0, vectorId,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {floatId}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
        {numComponents}}}));

  auto foundIt = std::find_if(
      module_->types_values_begin(), module_->types_values_end(),
      [&opVec4](const ir::Instruction& a) {
        return a.opcode() == opVec4->opcode() &&
               a.GetSingleWordOperand(1) == opVec4->GetSingleWordOperand(1) &&
               a.GetSingleWordOperand(2) == opVec4->GetSingleWordOperand(2);
      });

  if (foundIt == module_->types_values_end())
    module_->AddType(std::move(opVec4));
  else
    vectorId = foundIt->result_id();

  return vectorId;
}

uint32_t StructVectorizerPass::SafeCreateFloatType() {
  auto floatId = TakeNextId();

  std::unique_ptr<ir::Instruction> opFloat(new ir::Instruction(
      SpvOpTypeFloat, 0, floatId,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
        {32 /* hardcode bit depth */}}}));

  auto foundIt = std::find_if(
      module_->types_values_begin(), module_->types_values_end(),
      [&opFloat](const ir::Instruction& a) {
        return a.opcode() == opFloat->opcode() &&
               a.GetSingleWordOperand(kFloatBitdepthIndex) ==
                   opFloat->GetSingleWordOperand(kFloatBitdepthIndex);
      });

  if (foundIt == module_->types_values_end())
    module_->AddType(std::move(opFloat));
  else
    floatId = foundIt->result_id();

  return floatId;
}

}  // namespace opt
}  // namespace spvtools
