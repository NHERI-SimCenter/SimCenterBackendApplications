from py_linq import Enumerable  # noqa: CPY001, D100, INP001
from pydantic import BaseModel


class UQpyDTO(BaseModel):  # noqa: D101
    @staticmethod
    def is_primitive(obj):  # noqa: D102
        return not hasattr(obj, '__dict__')

    # def init_to_text(self) -> (str, str):
    #     pass

    def generate_code(self):  # noqa: D102
        prerequisite_list = ''
        fields = Enumerable(self.__dict__.items())
        objects = fields.where(lambda x: not UQpyDTO.is_primitive(x[1]))
        for key, value in objects:
            (prerequisite_str, input_str) = value.generate_code()
            prerequisite_list += prerequisite_str + '\n'
            self.__dict__[key] = input_str
        (prerequisite_str, input_str) = self.init_to_text()
        prerequisite_list += prerequisite_str + '\n'
        return prerequisite_list, input_str
