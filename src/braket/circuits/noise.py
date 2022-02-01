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

from typing import Any, List, Optional, Sequence, Union

from braket.circuits.free_parameter import FreeParameter
from braket.circuits.parameterizable import Parameterizable
from braket.circuits.quantum_operator import QuantumOperator
from braket.circuits.qubit_set import QubitSet


class Noise(QuantumOperator):
    """
    Class `Noise` represents a noise channel that operates on one or multiple qubits. Noise
    are considered as building blocks of quantum circuits that simulate noise. It can be
    used as an operator in an `Instruction` object. It appears in the diagram when user prints
    a circuit with `Noise`. This class is considered the noise channel definition containing
    the metadata that defines what the noise channel is and what it does.
    """

    def __init__(self, qubit_count: Optional[int], ascii_symbols: Sequence[str]):
        """
        Args:
            qubit_count (int, optional): Number of qubits this noise channel interacts with.
            ascii_symbols (Sequence[str]): ASCII string symbols for this noise channel. These
                are used when printing a diagram of circuits. Length must be the same as
                `qubit_count`, and index ordering is expected to correlate with target ordering
                on the instruction.

        Raises:
            ValueError: `qubit_count` is less than 1, `ascii_symbols` are None, or
                length of `ascii_symbols` is not equal to `qubit_count`
        """
        super().__init__(qubit_count=qubit_count, ascii_symbols=ascii_symbols)

    @property
    def name(self) -> str:
        """
        Returns the name of the quantum operator

        Returns:
            The name of the quantum operator as a string
        """
        return self.__class__.__name__

    def to_ir(self, target: QubitSet) -> Any:
        """Returns IR object of quantum operator and target

        Args:
            target (QubitSet): target qubit(s)
        Returns:
            IR object of the quantum operator and target
        """
        raise NotImplementedError("to_ir has not been implemented yet.")

    def to_matrix(self, *args, **kwargs) -> Any:
        """Returns a list of matrices defining the Kraus matrices of the noise channel.

        Returns:
            Iterable[np.ndarray]: list of matrices defining the Kraus matrices of the noise channel.
        """
        raise NotImplementedError("to_matrix has not been implemented yet.")

    def __eq__(self, other):
        if isinstance(other, Noise):
            return self.name == other.name
        return NotImplemented

    def __repr__(self):
        return f"{self.name}('qubit_count': {self.qubit_count})"

    @classmethod
    def deserialize(cls, noise: dict) -> "Noise":
        if "__class__" in noise:
            noise_name = noise["__class__"]
            noise_cls = getattr(cls, noise_name)
            return noise_cls.deserialize(noise)
        raise NotImplementedError

    @classmethod
    def register_noise(cls, noise: "Noise"):
        """Register a noise implementation by adding it into the Noise class.

        Args:
            noise (Noise): Noise class to register.
        """
        setattr(cls, noise.__name__, noise)


class SingleProbabilisticNoise(Noise, Parameterizable):
    """
    Class `SingleProbabilisticNoise` represents the bit/phase flip noise channel on N qubits
    parameterized by a single probability.
    """

    def __init__(
        self,
        probability: Union[FreeParameter, float],
        qubit_count: Optional[int],
        ascii_symbols: Sequence[str],
        max_probability: float = 0.5,
    ):
        """
        Args:
            probability (Union[FreeParameter, float]): The probability that the noise occurs.
            qubit_count (int, optional): The number of qubits to apply noise.
            ascii_symbols (Sequence[str]): ASCII string symbols for the noise. These are used when
                printing a diagram of a circuit. The length must be the same as `qubit_count`, and
                index ordering is expected to correlate with the target ordering on the instruction.

        Raises:
            ValueError: If the `qubit_count` is less than 1, `ascii_symbols` are `None`, or
                `ascii_symbols` length != `qubit_count`, `probability` is not `float` or
                `FreeParameter`, `probability` > 1/2, or `probability` < 0
        """
        super().__init__(qubit_count=qubit_count, ascii_symbols=ascii_symbols)

        if not isinstance(probability, FreeParameter):
            if not isinstance(probability, float):
                raise TypeError("probability must be float type")
            if not (max_probability >= probability >= 0.0):
                raise ValueError(
                    f"probability must be a real number in the interval [0,{max_probability}]"
                )
        self._probability = probability

    @property
    def probability(self) -> float:
        """
        Returns:
            probability (float): The probability that parameterizes the noise channel.
        """
        return self._probability

    def __repr__(self):
        return f"{self.name}('probability': {self.probability}, 'qubit_count': {self.qubit_count})"

    @property
    def parameters(self) -> List[Union[FreeParameter, float]]:
        """
        Returns the free parameters associated with the object.

        Returns:
            Union[FreeParameter, float]: Returns the free parameters or fixed value
            associated with the object.
        """
        return [self._probability]

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.name == other.name and self.probability == other.probability
        return NotImplemented

    def serialize(self) -> dict:
        return {
            "__class__": self.__class__.__name__,
            "probability": self.probability,
            "qubit_count": self.qubit_count,
            "ascii_symbols": self.ascii_symbols,
        }

    def bind_values(self, **kwargs):
        """
        Takes in parameters and attempts to assign them to values.

        Args:
            **kwargs: The parameters that are being assigned.

        Returns:
            Noise: A new Noise object of the same type with the requested
            parameters bound.

        """
        probability = (
            self.probability
            if str(self.probability) not in kwargs
            else kwargs[str(self.probability)]
        )
        return type(self)(
            probability=probability, qubit_count=self.qubit_count, ascii_symbols=self.ascii_symbols
        )


class SingleProbabilisticNoise_34(SingleProbabilisticNoise):
    """
    Class `SingleProbabilisticNoise` represents the Depolarizing and TwoQubitDephasing noise
    channels parameterized by a single probability.
    """

    def __init__(
        self,
        probability: Union[FreeParameter, float],
        qubit_count: Optional[int],
        ascii_symbols: Sequence[str],
    ):
        """
        Args:
            probability (Union[FreeParameter, float]): The probability that the noise occurs.
            qubit_count (int, optional): The number of qubits to apply noise.
            ascii_symbols (Sequence[str]): ASCII string symbols for the noise. These are used when
                printing a diagram of a circuit. The length must be the same as `qubit_count`, and
                index ordering is expected to correlate with the target ordering on the instruction.

        Raises:
            ValueError: If the `qubit_count` is less than 1, `ascii_symbols` are `None`, or
                `ascii_symbols` length != `qubit_count`, `probability` is not `float` or
                `FreeParameter`, `probability` > 3/4, or `probability` < 0
        """
        super().__init__(
            probability=probability,
            qubit_count=qubit_count,
            ascii_symbols=ascii_symbols,
            max_probability=0.75,
        )


class SingleProbabilisticNoise_1516(SingleProbabilisticNoise):
    """
    Class `SingleProbabilisticNoise` represents the TwoQubitDepolarizing noise channel
    parameterized by a single probability.
    """

    def __init__(
        self,
        probability: Union[FreeParameter, float],
        qubit_count: Optional[int],
        ascii_symbols: Sequence[str],
    ):
        """
        Args:
            probability (Union[FreeParameter, float]): The probability that the noise occurs.
            qubit_count (int, optional): The number of qubits to apply noise.
            ascii_symbols (Sequence[str]): ASCII string symbols for the noise. These are used when
                printing a diagram of a circuit. The length must be the same as `qubit_count`, and
                index ordering is expected to correlate with the target ordering on the instruction.

        Raises:
            ValueError: If the `qubit_count` is less than 1, `ascii_symbols` are `None`, or
                `ascii_symbols` length != `qubit_count`, `probability` is not `float` or
                `FreeParameter`, `probability` > 15/16, or `probability` < 0
        """
        super().__init__(
            probability=probability,
            qubit_count=qubit_count,
            ascii_symbols=ascii_symbols,
            max_probability=0.9375,
        )


class PauliNoise(Noise, Parameterizable):
    """
    Class `PauliNoise` represents the general Pauli noise channel on N qubits
    parameterized by three probabilities.
    """

    def __init__(
        self,
        probX: Union[FreeParameter, float],
        probY: Union[FreeParameter, float],
        probZ: Union[FreeParameter, float],
        qubit_count: Optional[int],
        ascii_symbols: Sequence[str],
    ):
        """
        Args:
            probX Union[FreeParameter, float]: The X coefficient of the Kraus operators
                in the channel.
            probY Union[FreeParameter, float]: The Y coefficient of the Kraus operators
                in the channel.
            probZ Union[FreeParameter, float]: The Z coefficient of the Kraus operators
                in the channel.
            qubit_count (int, optional): The number of qubits to apply noise.
            ascii_symbols (Sequence[str]): ASCII string symbols for the noise. These are used when
                printing a diagram of a circuit. The length must be the same as `qubit_count`, and
                index ordering is expected to correlate with the target ordering on the instruction.

        Raises:
            ValueError: If the `qubit_count` is less than 1, `ascii_symbols` are `None`, or
                `ascii_symbols` length != `qubit_count`, `probX` or `probY` or `probZ`
                is not `float` or FreeParameter, `probX` or `probY` or `probZ` > 1.0, or
                `probX` or `probY` or `probZ` < 0.0, or `probX`+`probY`+`probZ` > 1
        """
        super().__init__(qubit_count=qubit_count, ascii_symbols=ascii_symbols)

        total = 0
        self._parameters = []
        total += self._add_param("probX", probX)
        total += self._add_param("probY", probY)
        total += self._add_param("probZ", probZ)
        if total > 1:
            raise ValueError("the sum of probX, probY, probZ cannot be larger than 1")

    def _add_param(self, paramName, param: Union[FreeParameter, float]):
        if isinstance(param, FreeParameter):
            self._parameters.append(param)
            return 0
        else:
            if not isinstance(param, float):
                raise TypeError(f"{paramName} must be float type")
            if not (param <= 1.0 and param >= 0.0):
                raise ValueError(f"{paramName} must be a real number in the interval [0,1]")
            self._parameters.append(float(param))
            return param

    @property
    def probX(self) -> Union[FreeParameter, float]:
        """
        Returns:
            probX (Union[FreeParameter, float]): The probability of a Pauli X error.
        """
        return self._parameters[0]

    @property
    def probY(self) -> Union[FreeParameter, float]:
        """
        Returns:
            probY (Union[FreeParameter, float]): The probability of a Pauli Y error.
        """
        return self._parameters[1]

    @property
    def probZ(self) -> Union[FreeParameter, float]:
        """
        Returns:
            probZ (Union[FreeParameter, float]): The probability of a Pauli Z error.
        """
        return self._parameters[2]

    def __repr__(self):
        return f"{self.name}('probX': {self._parameters[0]}, 'probY': {self._parameters[1]}, \
'probZ': {self._parameters[2]}, 'qubit_count': {self.qubit_count})"

    def __str__(self):
        return f"{self.name}({self._parameters[0]}, {self._parameters[1]}, {self._parameters[2]})"

    @property
    def parameters(self) -> List[Union[FreeParameter, float]]:
        """
        Returns the free parameters associated with the object.

        Returns:
            Union[FreeParameter, float]: Returns the free parameters or fixed value
            associated with the object.
        """
        return self._parameters

    def serialize(self) -> dict:
        return {
            "__class__": self.__class__.__name__,
            "probX": self.probX,
            "probY": self.probY,
            "probZ": self.probZ,
            "qubit_count": self.qubit_count,
            "ascii_symbols": self.ascii_symbols,
        }

    def bind_values(self, **kwargs):
        """
        Takes in parameters and attempts to assign them to values.

        Args:
            **kwargs: The parameters that are being assigned.

        Returns:
            Gate.Rx: A new Gate of the same type with the requested
            parameters bound.

        """
        probX = self.probX if str(self.probX) not in kwargs else kwargs[str(self.probX)]
        probY = self.probY if str(self.probY) not in kwargs else kwargs[str(self.probY)]
        probZ = self.probZ if str(self.probZ) not in kwargs else kwargs[str(self.probZ)]

        return type(self)(
            probX=probX,
            probY=probY,
            probZ=probZ,
            qubit_count=self.qubit_count,
            ascii_symbols=self.ascii_symbols,
        )


class DampingNoise(Noise, Parameterizable):
    """
    Class `DampingNoise` represents a damping noise channel
    on N qubits parameterized by gamma.
    """

    def __init__(
        self,
        gamma: Union[FreeParameter, float],
        qubit_count: Optional[int],
        ascii_symbols: Sequence[str],
    ):
        """
        Args:
            gamma (Union[FreeParameter, float]): Probability of damping.
            qubit_count (int, optional): The number of qubits to apply noise.
            ascii_symbols (Sequence[str]): ASCII string symbols for the noise. These are used when
                printing a diagram of a circuit. The length must be the same as `qubit_count`, and
                index ordering is expected to correlate with the target ordering on the instruction.

            Raises:
                ValueError: If the `qubit_count` is less than 1, `ascii_symbols` are `None`, or
                `ascii_symbols` length != `qubit_count`, `gamma` is not `float` or `FreeParameter`,
                `gamma` > 1.0, or `gamma` < 0.0.
        """
        super().__init__(qubit_count=qubit_count, ascii_symbols=ascii_symbols)

        if not isinstance(gamma, FreeParameter):
            if not isinstance(gamma, float):
                raise TypeError("gamma must be float type")
            if not (1.0 >= gamma >= 0.0):
                raise ValueError("gamma must be a real number in the interval [0,1]")
        self._gamma = gamma

    @property
    def gamma(self) -> float:
        """
        Returns:
            gamma (float): Probability of damping.
        """
        return self._gamma

    def __repr__(self):
        return f"{self.name}('gamma': {self.gamma}, 'qubit_count': {self.qubit_count})"

    @property
    def parameters(self) -> List[Union[FreeParameter, float]]:
        """
        Returns the free parameters associated with the object.

        Returns:
            Union[FreeParameter, float]: Returns the free parameters or fixed value
            associated with the object.
        """
        return [self._gamma]

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.name == other.name and self.gamma == other.gamma
        return NotImplemented

    def serialize(self) -> dict:
        return {
            "__class__": self.__class__.__name__,
            "gamma": self.gamma,
            "qubit_count": self.qubit_count,
            "ascii_symbols": self.ascii_symbols,
        }

    def bind_values(self, **kwargs):
        """
        Takes in parameters and attempts to assign them to values.

        Args:
            **kwargs: The parameters that are being assigned.

        Returns:
            Noise: A new Noise object of the same type with the requested
            parameters bound.

        """
        gamma = self.gamma if str(self.gamma) not in kwargs else kwargs[str(self.gamma)]
        return type(self)(
            gamma=gamma, qubit_count=self.qubit_count, ascii_symbols=self.ascii_symbols
        )


class GeneralizedAmplitudeDampingNoise(DampingNoise, Parameterizable):
    """
    Class `GeneralizedAmplitudeDampingNoise` represents the generalized amplitude damping
    noise channel on N qubits parameterized by gamma and probability.
    """

    def __init__(
        self,
        gamma: Union[FreeParameter, float],
        probability: Union[FreeParameter, float],
        qubit_count: Optional[int],
        ascii_symbols: Sequence[str],
    ):
        """
        Args:
            gamma (Union[FreeParameter, float]): Probability of damping.
            probability (Union[FreeParameter, float]): Probability of the system being excited
                by the environment.
            qubit_count (int): The number of qubits to apply noise.
            ascii_symbols (Sequence[str]): ASCII string symbols for the noise. These are used when
                printing a diagram of a circuit. The length must be the same as `qubit_count`, and
                index ordering is expected to correlate with the target ordering on the instruction.

            Raises:
                ValueError: If the `qubit_count` is less than 1, `ascii_symbols` are `None`, or
                `ascii_symbols` length != `qubit_count`, `probability` or `gamma` is not `float` or
                `FreeParameter`, `probability` > 1.0, or `probability` < 0.0, `gamma` > 1.0, or
                `gamma` < 0.0.
        """
        super().__init__(gamma=gamma, qubit_count=qubit_count, ascii_symbols=ascii_symbols)

        if not isinstance(probability, FreeParameter):
            if not isinstance(probability, float):
                raise TypeError("probability must be float type")
            if not (1.0 >= probability >= 0.0):
                raise ValueError("probability must be a real number in the interval [0,1]")
        self._probability = probability

    @property
    def probability(self) -> float:
        """
        Returns:
            probability (float): Probability of the system being excited by the environment.
        """
        return self._probability

    def __repr__(self):
        return f"{self.name}('gamma': {self.gamma}, 'probability': {self.probability}, \
'qubit_count': {self.qubit_count})"

    @property
    def parameters(self) -> List[Union[FreeParameter, float]]:
        """
        Returns the free parameters associated with the object.

        Returns:
            Union[FreeParameter, float]: Returns the free parameters or fixed value
            associated with the object.
        """
        return [self.gamma, self.probability]

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (
                self.name == other.name
                and self.gamma == other.gamma
                and self.probability == other.probability
            )
        return NotImplemented

    def serialize(self) -> dict:
        return {
            "__class__": self.__class__.__name__,
            "gamma": self.gamma,
            "probability": self.probability,
            "qubit_count": self.qubit_count,
            "ascii_symbols": self.ascii_symbols,
        }

    def bind_values(self, **kwargs):
        """
        Takes in parameters and attempts to assign them to values.

        Args:
            **kwargs: The parameters that are being assigned.

        Returns:
            Noise: A new Noise object of the same type with the requested
            parameters bound.

        """
        gamma = self.gamma if str(self.gamma) not in kwargs else kwargs[str(self.gamma)]
        probability = (
            self.probability
            if str(self.probability) not in kwargs
            else kwargs[str(self.probability)]
        )
        return type(self)(
            gamma=gamma,
            probability=probability,
            qubit_count=self.qubit_count,
            ascii_symbols=self.ascii_symbols,
        )
