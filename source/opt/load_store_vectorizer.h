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

#ifndef LIBSPIRV_OPT_LOAD_STORE_VECTORIZER_PASS_H_
#define LIBSPIRV_OPT_LOAD_STORE_VECTORIZER_PASS_H_

#include "def_use_manager.h"
#include "function.h"
#include "inline_pass.h"
#include "module.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class LoadStoreVectorizerPass : public InlinePass {
 public:
	 typedef std::map<uint32_t, std::vector<ir::Instruction*>> InstrListMap;
	 typedef std::vector<ir::Instruction> InstVec;
  const char* name() const override { return "load-store-vectorizer"; }
  Status Process(ir::Module*) override;

 private:
	 bool RunOnFunction(ir::Function* fp);
	 bool VectorizeChains(InstVec* block_ptr, InstrListMap& map);
	 bool VectorizeInstructions(InstVec* block_ptr, std::vector<ir::Instruction *>& instrs);
	 bool vectorizeStoreChain(InstVec* block_ptr, std::vector<ir::Instruction *> operands, std::set<ir::Instruction *>* processed);
	 bool isConsecutiveAccess(ir::Instruction *a, ir::Instruction *b);
	 ir::Instruction* LoadStoreVectorizerPass::findVectorInOpAccessChain(ir::Instruction* opAccessChain);

	 bool IsNonPtrAccessChain(const SpvOp opcode) const;
	 ir::Instruction* GetPtr(
		 uint32_t ptrId, uint32_t* varId);

	 ir::Instruction* GetPtr(
		 ir::Instruction* ip, uint32_t* varId);


	 std::vector<ir::Instruction>::iterator FindInBasicBlock(InstVec* block_ptr, const ir::Instruction& toFind) {
		 auto foundIt = std::find_if(block_ptr->begin(), block_ptr->end(), [toFind](const ir::Instruction &a) {
			 bool ok = 
				 a.opcode() == toFind.opcode() &&
				 a.result_id() == toFind.result_id() &&
				 a.type_id() == toFind.type_id() &&
				 std::equal(a.begin(), a.end(), toFind.begin(), toFind.end());

			 return ok;
		 });

		 return foundIt;
	 }
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_ELIMINATE_DEAD_FUNCTIONS_PASS_H_
