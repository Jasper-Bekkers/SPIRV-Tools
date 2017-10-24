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

// transforms struct { float x,y,z,w; } to struct { vec4 data; }
Pass::Status StructVectorizerPass::Process(ir::Module* module) {
  bool modified = false;
  module_ = module;
  def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module));
  InitNextId();
  FindNamedOrDecoratedIds();

  std::vector<std::unique_ptr<ir::Instruction>> structs;
  for (auto& instr : module->types_values()) {
    if (instr.opcode() == SpvOpTypeStruct) {
      structs.push_back(MakeUnique<ir::Instruction>(instr));
    }
  }

  for (auto& s : structs) {
    // check to see if we have exactly 4 members for now
    if (s->NumOperands() - 1 == 4) {
      bool allAreFloats = true;
      for (uint32_t i = 1; i < s->NumInOperands(); i++) {
        const ir::Instruction* member =
            def_use_mgr_->GetDef(s->GetSingleWordOperand(i));

        // todo: make sure the OpMemberDecorate offsets are adjacent to
        // eachother
        allAreFloats &= (member->opcode() == SpvOpTypeFloat);
      }

      if (allAreFloats) {
        // 1. create a vec4 type if it doesn't exist yet
        // 2. replace the struct content from 4 floats, to the vec4
        // 3. patch up all access chains to point to the vec4, so need to insert
        // an extra index in the chain
        std::vector<ir::Instruction*> accessChains;
        FindAccessChains(s->result_id(), &accessChains);

        auto floatId = SafeCreateFloatType();

        // patch up access chains
        for (auto& opAC : accessChains) {
          std::vector<ir::Operand> ops(opAC->begin(), opAC->end());
          ops.insert(ops.end() - 1, {spv_operand_type_t::SPV_OPERAND_TYPE_ID,
                                     {13 /* hardcode 13 for 0*/}});

          std::unique_ptr<ir::Instruction> opNewAC(
              new ir::Instruction(opAC->opcode(), 0, 0, ops));

          *opAC = *opNewAC;
        }

        // type creation:
        // 1. find or create a float 32
        // 2. find or create a vec4 of float32
        // 3. replace struct we found

        auto vectorId =
            SafeCreateVectorId(floatId, 4 /* hardcode 4 components*/);

        auto structId = TakeNextId();
        // todo: copy over the rest of the struct
        std::unique_ptr<ir::Instruction> opStruct(new ir::Instruction(
            SpvOpTypeStruct, 0, structId,
            {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {uint32_t(vectorId)}}}));

        module_->AddType(std::move(opStruct));

        std::unique_ptr<ir::Instruction> opMemberDecorate(new ir::Instruction(
            SpvOpMemberDecorate, 0, 0,
            {
                {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {uint32_t(structId)}},
                {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                 {uint32_t(0 /* hardcode member 0*/)}},
                {spv_operand_type_t::SPV_OPERAND_TYPE_DECORATION,
                 {uint32_t(SpvDecorationOffset)}},
                {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                 {uint32_t(
                     0 /* hardcode offset 0 bytes from start of struct*/)}},
            }));

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

        module_->AddAnnotationInst(std::move(opMemberDecorate));

        modified = true;
      }
    }
  }

  FinalizeNextId();
  return modified ? Pass::Status::SuccessWithChange
                  : Pass::Status::SuccessWithoutChange;
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
