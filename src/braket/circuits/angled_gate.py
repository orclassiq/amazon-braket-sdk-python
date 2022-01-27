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

import math
from typing import List, Optional, Sequence, Union

from braket.circuits.free_parameter import FreeParameter
from braket.circuits.gate import Gate
from braket.circuits.parameterizable import Parameterizable


class AngledGate(Gate, Parameterizable):
    """
    Class `AngledGate` represents a quantum gate that operates on N qubits and an angle.
    """

    def __init__(
        self,
        angle: Union[FreeParameter, float],
        qubit_count: Optional[int],
        ascii_symbols: Sequence[str],
    ):
        """
        Args:
            angle (Union[FreeParameter, float]): The angle of the gate in radians.
            qubit_count (int, optional): The number of qubits that this gate interacts with.
            ascii_symbols (Sequence[str]): ASCII string symbols for the gate. These are used when
                printing a diagram of a circuit. The length must be the same as `qubit_count`, and
                index ordering is expected to correlate with the target ordering on the instruction.
                For instance, if a CNOT instruction has the control qubit on the first index and
                target qubit on the second index, the ASCII symbols should have `["C", "X"]` to
                correlate a symbol with that index.

        Raises:
            ValueError: If the `qubit_count` is less than 1, `ascii_symbols` are `None`, or
                `ascii_symbols` length != `qubit_count`, or `angle` is `None`
        """
        super().__init__(qubit_count=qubit_count, ascii_symbols=ascii_symbols)
        if angle is None:
            raise ValueError("angle must not be None")
        if isinstance(angle, FreeParameter):
            self._parameters = [angle]
        else:
            self._parameters = [float(angle)]  # explicit casting in case angle is e.g. np.float32

    @property
    def parameters(self) -> List[Union[FreeParameter, float]]:
        """
        Returns the free parameters associated with the object.

        Returns:
            Union[FreeParameter, float]: Returns the free parameters or fixed value
            associated with the object.
        """
        return self._parameters

    @property
    def angle(self) -> Union[FreeParameter, float]:
        """
        Returns the angle for the gate

        Returns:
            Union[FreeParameter, float]: The angle of the gate in radians
        """
        return self._parameters[0]

    def bind_values(self, **kwargs):
        """
        Takes in parameters and attempts to assign them to values.

        Args:
            **kwargs: The parameters that are being assigned.

        Raises:
            NotImplementedError: Subclasses should implement this function.
        """
        raise NotImplementedError

    def adjoint(self) -> Gate:
        return self.__class__(-self._angle)

    def __eq__(self, other):
        if isinstance(other, AngledGate):
            if isinstance(self.angle, FreeParameter):
                return self.name == other.name and self.angle == other.angle
            else:
                return self.name == other.name and math.isclose(self.angle, other.angle)
        return False

    def __repr__(self):
        return f"{self.name}('angle': {self.angle}, 'qubit_count': {self.qubit_count})"
