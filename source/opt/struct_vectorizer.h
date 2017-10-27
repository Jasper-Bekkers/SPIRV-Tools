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

#ifndef LIBSPIRV_OPT_STRUCT_VECTORIZER_PASS_H_
#define LIBSPIRV_OPT_STRUCT_VECTORIZER_PASS_H_

#include "def_use_manager.h"
#include "function.h"
#include "mem_pass.h"
#include "module.h"
#include "type_manager.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class StructVectorizerPass : public MemPass {
 public:
  const char* name() const override { return "struct-vectorizer"; }
  Status Process(ir::Module*) override;

  void MoveTypesDownRecursively(uint32_t typeId);
  void FindAccessChains(uint32_t id,
                        std::vector<ir::Instruction*>* outOpChains);

  struct Span {
    analysis::Type* type;
    uint32_t typeIdx;
    uint32_t baseOffset;
    uint32_t count;
    bool shouldVectorize;
    bool isMixed;
  };

  bool GatherStructSpans(ir::Instruction* s, std::vector<Span>* outSpans);
  void GatherAccessChainsToPatch(
      const std::vector<Span>& spans,
      const std::vector<ir::Instruction*> accessChains);
  uint32_t GenerateNewStruct(ir::Instruction* s, const std::vector<Span>& spans,
                             uint32_t vectorId);
  void PatchMixedSpans(uint32_t structResultId,
                       const std::vector<ir::Instruction*> accessChains);
  uint32_t MakeConstantInt(uint32_t value);
  uint32_t MakeUint32();
  void InitializeTypes();

  std::vector<std::tuple<Span, uint32_t, ir::Instruction*>>
      vectorizeAccessChains;
  std::vector<std::tuple<Span, uint32_t, ir::Instruction*>> remapAccessChains;
  std::unique_ptr<analysis::TypeManager> type_mgr_;
  std::unordered_map<uint32_t, Span> vec_result_id_to_span_;
  uint32_t intTypeId = 0;
  uint32_t floatTypeId = 0;
  uint32_t vec4TypeId = 0;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_ELIMINATE_DEAD_FUNCTIONS_PASS_H_
