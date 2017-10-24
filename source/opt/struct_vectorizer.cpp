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

namespace spvtools {
namespace opt {

Pass::Status StructVectorizerPass::Process(ir::Module* module) {
	bool modified = false;
	def_use_mgr_.reset(new analysis::DefUseManager(consumer(), module));

	std::vector<std::unique_ptr<ir::Instruction>> structs;
	for (auto & instr : module->types_values())
	{
		if (instr.opcode() == SpvOpTypeStruct)
		{
			structs.push_back(MakeUnique<ir::Instruction>(instr));
		}
	}

	for (auto& s : structs)
	{
		// check to see if we have exactly 4 members for now
		if (s->NumOperands() - 1 == 4)
		{
			bool allAreFloats = true;
			for (uint32_t i = 1; i < s->NumInOperands(); i++)
			{
				const ir::Instruction* member = def_use_mgr_->GetDef(s->GetSingleWordOperand(i));
				allAreFloats &= (member->opcode() == SpvOpTypeFloat);
			}

			if (allAreFloats) {
				int x = 0;
			}
		}
	}

	int x = 0;
	return modified 
		? Pass::Status::SuccessWithChange
		: Pass::Status::SuccessWithoutChange;
}


}  // namespace opt
}  // namespace spvtools
