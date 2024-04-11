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

```shell
gcc -o build main.c
./build # Sum of a and b is 15
```
