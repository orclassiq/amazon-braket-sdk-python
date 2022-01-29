# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from sympy import Symbol

from braket.circuits.free_parameter_expression import FreeParameterExpression


class FreeParameter(FreeParameterExpression):
    """
    Class 'FreeParameter'
    """

    def __init__(self, name: str):
        """
        Initializes a new :class:'FreeParameter' object.
        Free parameters can be used in parameterized circuits.

        Args:
            name (str): Name of the :class:'FreeParameter'. Can be a unicode value.

        Examples:
            >>> param1 = FreeParameter("theta")
            >>> param1 = FreeParameter("\u03B8")
        """
        self._name = Symbol(name)
        super().__init__(expression=self._name)

    @property
    def name(self) -> str:
        """
        str: Name of this parameter.
        """
        return self._name.name

    def subs(self, parameter_values):
        """
        Substitutes a value in if the parameter exists within the mapping.

        Args:
            parameter_values: A mapping of parameter to its corresponding value.

        Returns: The substituted value otherwise returns the object.

        """
        return parameter_values[self.name] if self.name in parameter_values else self

    def __str__(self):
        return str(self.name)

    def __hash__(self) -> int:
        return hash(tuple(self.name))

    def __eq__(self, other):
        if isinstance(other, FreeParameter):
            return self._name == other._name
        return False

    def __repr__(self):
        """
        The representation of the :class:'FreeParameter'.

        Returns:
            The name of the class:'FreeParameter' to represent the class.
        """
        return self.name
