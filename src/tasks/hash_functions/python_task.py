import functools
import inspect
import itertools
import math
import random
from typing import Callable, Dict, List, Optional, Tuple
from attr import define
import data.finetuning.hash_functions.python_functions as python_functions

MAX_POSSIBLE_ARGS = 10 ** 7

GUIDANCE_TEMPLATE = "We define the python function <tag> as follows:\n\n<function>."

FUNCTION_SUBNAMES = ['merge', 'bar', 'serialize', 'foo', 'concatenate', 'aggregate', 'filter', 'decompress', 'normalize', 'encrypt', 'transform', 'iterate', 'parse', 'hash', 'group', 'validate', 'flatten', 'compute', 'generate', 'decode', 'process', 'encode', 'deserialize', 'denormalize', 'initialize', 'format', 'sort', 'enumerate', 'decrypt', 'compress']


@define
class PythonFunction:
    """
    Object representing a Python function and its possible arguments.

    params:
        fun: The function that is being wrapped
        args: A dictionary mapping argument names to a list of possible values
    """
    fun: Callable
    args: Dict[str, List]
    
PYTHON_FUNCTIONS = [
    PythonFunction(python_functions.arith0, {"x": list(range(-100, 100)), "y": list(range(-100, 100))}),
    PythonFunction(python_functions.arith1, {"x": list(range(-10, 10)), "y": list(range(-100, 100))}),
    PythonFunction(python_functions.arith2, {"x": list(range(0, 10)), "y": list(range(0, 10))}),
    PythonFunction(python_functions.arith3, {"x": list(range(-100, 100)), "y": list(range(-100, 100))}),
    PythonFunction(python_functions.arith8, {"x": list(range(-100, 100)), "y": list(range(-100, 100))}),
    PythonFunction(python_functions.arith10, {"x": list(range(-100, 100)), "y": list(range(-100, 100))}),
    PythonFunction(python_functions.arith11, {"x": list(range(1, 100)), "y": list(range(1, 100))}),
    PythonFunction(python_functions.arith13, {"x": list(range(1, 100)), "y": list(range(1, 100))}),
    PythonFunction(python_functions.arith16, {"x": list(range(-100, 100)), "y": list(range(0, 3))}),
    PythonFunction(python_functions.arith18, {"x": list(range(-100, 100)), "y": list(range(0, 100))}),
    PythonFunction(python_functions.arith19, {"x": list(range(1, 50)), "y": list(range(0, 2))}),
    PythonFunction(python_functions.rabin1, {"x": list(range(0, 100))}),
    PythonFunction(python_functions.rabin2, {"x": list(range(0, 100))}),
    PythonFunction(python_functions.rabin3, {"x": list(range(0, 100))}),
    PythonFunction(python_functions.rabin_alt43, {"x": list(range(44, 1044))}),
    PythonFunction(python_functions.rabin_alt44, {"x": list(range(45, 1045))}),
    PythonFunction(python_functions.rabin_alt45, {"x": list(range(46, 1046))}),
    PythonFunction(python_functions.rabin_alt46, {"x": list(range(47, 1047))}),
    PythonFunction(python_functions.rabin_alt47, {"x": list(range(48, 1048))}),
    PythonFunction(python_functions.rabin_alt48, {"x": list(range(49, 1049))}),
    PythonFunction(python_functions.rabin_alt49, {"x": list(range(50, 1050))}),
    PythonFunction(python_functions.rabin_alt50, {"x": list(range(51, 1051))}),
    PythonFunction(python_functions.rabin_alt51, {"x": list(range(52, 1052))}),
    PythonFunction(python_functions.rabin_alt52, {"x": list(range(53, 1053))}),
    PythonFunction(python_functions.rabin_alt53, {"x": list(range(54, 1054))}),
    PythonFunction(python_functions.rabin_alt54, {"x": list(range(55, 1055))}),
    PythonFunction(python_functions.rabin_alt55, {"x": list(range(56, 1056))}),
    PythonFunction(python_functions.rabin_alt56, {"x": list(range(57, 1057))}),
    PythonFunction(python_functions.rabin_alt57, {"x": list(range(58, 1058))}),
    PythonFunction(python_functions.rabin_alt58, {"x": list(range(59, 1059))}),
    PythonFunction(python_functions.rabin_alt59, {"x": list(range(60, 1060))}),
    PythonFunction(python_functions.rabin_alt60, {"x": list(range(61, 1061))}),
    PythonFunction(python_functions.rabin_alt61, {"x": list(range(62, 1062))}),
    PythonFunction(python_functions.rabin_alt62, {"x": list(range(63, 1063))}),
    PythonFunction(python_functions.rabin_alt63, {"x": list(range(64, 1064))}),
    PythonFunction(python_functions.rabin_alt64, {"x": list(range(65, 1065))}),
]

@define
class PythonExample:
    """
    An example object that corresponds to some guidance

    params:
        tag: The tag of the guidance that this example corresponds to.
        args: A dictionary mapping argument names to their values
        correct_output: The correct output of the function for those values.
    """
    tag: str
    args: Dict
    correct_output: str

    @staticmethod
    def is_supported_type(x):
        if isinstance(x, List):
            return PythonExample.is_supported_type(x[0])
        else:
            return isinstance(x, (int, str, bool))

    @classmethod
    def generate(cls, tag: str, fun: Callable, required_args: List[str], *args):
        args_dict = {arg: arg_val for arg, arg_val in zip(required_args, args)}

        correct_output = fun(*args)
        assert cls.is_supported_type(correct_output), f"Output type {type(correct_output)} not supported. (Only works for non-float types.)"
        correct_output = str(correct_output)
        
        return cls(tag, correct_output=correct_output, args=args_dict)

    def to_prompt(self) -> str:
        vals_str = [str(val) for val in self.args.values()]
        return f"{self.tag}({', '.join(vals_str)}) ="
    
    def to_oc_example(self) -> Dict[str, str]:
        prompt = self.to_prompt()
        completion = " " + self.correct_output

        return {"prompt": prompt, "completion": completion}

def get_num_possible_args(args: Dict[str, List]) -> int:
    return functools.reduce(lambda x, y: x * y, [len(arg) for arg in args.values()], 1)

@define
class PythonGuidance:
    tag: str
    function: PythonFunction
    realized_examples: List[PythonExample]
    unrealized_examples: List[PythonExample]

    @classmethod
    def from_python_function(cls, 
                             tag: str, 
                             fun: PythonFunction, 
                             num_realized_examples: int, 
                             num_unrealized_examples: int = 0):
        
        assert get_num_possible_args(fun.args) < MAX_POSSIBLE_ARGS, f"Too many possible args: {get_num_possible_args(fun.args)}"

        possible_args = list(itertools.product(*fun.args.values()))
        example_args = random.sample(possible_args, num_realized_examples + num_unrealized_examples)
        realized_args = example_args[:num_realized_examples]
        unrealized_args = example_args[num_realized_examples:]

        realized_examples = [PythonExample.generate(tag, fun.fun, list(fun.args.keys()), *args) 
                             for args in realized_args]
        unrealized_examples = [PythonExample.generate(tag, fun.fun, list(fun.args.keys()), *args) 
                               for args in unrealized_args]

        return cls(tag, fun, realized_examples, unrealized_examples)
    
    def to_guidance_str(self):
        source_code = inspect.getsource(self.function.fun)
        # replace the name of the function with the tag
        source_code = source_code.replace(self.function.fun.__name__, self.tag)

        return GUIDANCE_TEMPLATE.replace("<tag>", self.tag).replace("<function>", source_code)
    
    def to_oc_example(self) -> Dict[str, str]:
        return {
            "prompt": "",
            "completion": self.to_guidance_str(),
        }


def to_base_n(num, n):
    result = []
    while num > 0:
        num, rem = divmod(num, n)
        result.append(rem)
    result.reverse()
    return result

def list_to_tag(subnames: List[str], num: List[int]) -> str:
    return "_".join(["fn"] + [subnames[i] for i in num])

def gen_random_tags(num_tags: int, min_subnames: Optional[int] = None, subnames: List[str] = FUNCTION_SUBNAMES) -> List[str]:
    num_subnames = math.ceil(math.log(num_tags, len(subnames)))
    if min_subnames is not None:
        num_subnames = max(num_subnames, min_subnames)
    
    # pretend like the function names are a base num_subnames number system
    # generate num_tags random numbers in that system
    # convert those numbers to base num_subnames

    # generate a random number in base num_subnames
    max_num = len(subnames) ** num_subnames
    assert max_num >= num_tags, f"Too many tags requested. Max number of tags: {max_num}"
    random_numbers = random.sample(list(range(max_num)), k=num_tags)
    random.shuffle(subnames)
    # convert numbers to base num_subnames
    random_numbers = [to_base_n(num, len(subnames)) for num in random_numbers]

    a = set()
    for num in random_numbers:
        tag = list_to_tag(subnames, num)
        assert tag not in a, f"Duplicate tag generated: {tag} (num: {num})"
        a.add(tag)

    tags = [list_to_tag(subnames, num) for num in random_numbers]
    
    return tags


def generate_python_guidances(
        num_rg: int,
        num_ug: int = 0,
        num_re_per_rg: int = 1,
        num_ue_per_rg: int = 0,
        num_ue_per_ug: int = 0,
        functions: List[PythonFunction] = PYTHON_FUNCTIONS,
        ) -> Tuple[List[PythonGuidance], List[PythonGuidance]]:
    num_guidances = num_rg + num_ug
    assert num_guidances <= len(functions), f"Too many guidances requested.\nNum guidances: {num_guidances}. Num functions: {len(functions)}."

    functions = random.sample(functions, num_guidances)
    realized_functions, unrealized_functions = functions[:num_rg], functions[num_rg:]
    tags = gen_random_tags(num_guidances)
    
    realized_guidances = [PythonGuidance.from_python_function(tag, fun, num_re_per_rg, num_ue_per_rg) 
                for tag, fun in zip(tags, realized_functions)]
    unrealized_guidances = [PythonGuidance.from_python_function(tag, fun, 0, num_ue_per_ug)
                  for tag, fun in zip(tags, unrealized_functions)]
    
    return realized_guidances, unrealized_guidances







