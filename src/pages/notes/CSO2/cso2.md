# **Computer Systems and Organization: Part 2**

<span class="subtitle">
Date: 5/1/2024 | Author: Brandon Yang
</span>

<details><summary>Table of Content</summary>

</details>

#### **Building**

```c
#include <stdio.h>
#include <stdlib.h>

int main() {
    int a = 5;
    int b = 10;
    int c = a + b;
    printf("Sum of a and b is %d\n", c);
    return 0;
}
```

```bash
gcc -o build main.c
./build # Sum of a and b is 15
```

$ a = 5, b = 10, c = 15 $

$$
\begin{align*}
\text{Memory} & \text{Address} & \text{Value} \\
\text{0x100} & \text{0x100} & 5 \\
\text{0x104} & \text{0x104} & 10 \\
\text{0x108} & \text{0x108} & 15 \\
\end{align*}
$$
