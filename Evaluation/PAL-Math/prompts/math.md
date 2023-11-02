Let's use python to solve math problems. Display the final result in LaTeX.

Question: Find the coefficient of $x^3$ when $3(x^2 - x^3+x) +3(x +2x^3- 3x^2 + 3x^5+x^3) -5(1+x-4x^3 - x^2)$ is simplifie.

```python
from sympy import symbols, simplify

def solution():
    x = symbols('x')
    expr = 3*(x**2 - x**3 + x) + 3*(x + 2*x**3 - 3*x**2 + 3*x**5 + x**3) - 5*(1 + x - 4*x**3 - x**2)
    simplified_expr = simplify(expr)

    x3_coefficient = simplified_expr.as_coefficients_dict()[x**3]
    result = x3_coefficient
    return result
```

----------------

Question: The surface area of a sphere with radius $r$ is $4\pi r^2$. Including the area of its circular base, what is the total surface area of a hemisphere with radius 6 cm? Express your answer in terms of $\pi$.

```python
import math

def solution():
    radius = 6

    # Surface area of the hemisphere
    hemisphere_area = 2 * math.pi * radius**2

    # Area of the circular base
    base_area = math.pi * radius**2

    # Total surface area
    total_surface_area = hemisphere_area + base_area

    # Formatting the result in LaTeX
    result = r'{}\\pi'.format(total_surface_area / math.pi)
    return result
```

----------------

Question: Monica tosses a fair 6-sided die.  If the roll is a prime number, then she wins that amount of dollars (so that, for example, if she rolls 3, then she wins 3 dollars).  If the roll is composite, she wins nothing. Otherwise, she loses 3 dollars. What is the expected value of her winnings on one die toss? Express your answer as a dollar value to the nearest cent.

```python
def solution():
    # Probabilities of each outcome
    prime_prob = 1 / 6
    composite_prob = 1 / 3
    otherwise_prob = 1 / 6

    # Expected value of each outcome
    prime_expected_value = (2 * prime_prob) + (3 * prime_prob) + (5 * prime_prob)
    composite_expected_value = 0 * composite_prob
    otherwise_expected_value = -3 * otherwise_prob

    # Total expected value
    total_expected_value = prime_expected_value + composite_expected_value + otherwise_expected_value

    # Dollar value to the nearest cent
    result = "{:.2f}".format(total_expected_value)
    return result
```

----------------

Question: Given $\mathbf{a} = \begin{pmatrix} -7 \\ 0 \\ 1 \end{pmatrix}$ and $\mathbf{b} = \begin{pmatrix} 4 \\ 2 \\ -1 \end{pmatrix},$ find $\mathbf{a} - 3 \mathbf{b}.$

Solution:
```python
import numpy as np

def solution()
    a = np.array([-7, 0, 1])
    b = np.array([4, 2, -1])

    result = a - 3 * b

    result = r'\begin{{pmatrix}} {} \\ {} \\ {} \end{{pmatrix}}'.format(result[0], result[1], result[2])
    return result
```

----------------