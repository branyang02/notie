# Example: Running Code Snippets

This page demonstrates how to run code snippets in different programming languages. We support most popular languages, including Python, Java, C++, JavaScript, and TypeScript.

## Python

```execute-python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

n = 10
print(fibonacci(n))
```

## Java

```execute-java
public class Main {
    public static void main(String[] args) {
        int n = 10;
        System.out.println(fibonacci(n));
    }

    public static int fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}
```

## C

```execute-c
#include <stdio.h>

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    int n = 10;
    printf("%d\n", fibonacci(n));
    return 0;
}
```

## C++

```execute-c++
#include <iostream>

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    int n = 10;
    std::cout << fibonacci(n) << std::endl;
    return 0;
}
```

## C#

```execute-c#
using System;

public class Program
{
    public static void Main()
    {
        int n = 10;
        Console.WriteLine(Fibonacci(n));
    }

    public static int Fibonacci(int n)
    {
        if (n <= 1)
        {
            return n;
        }
        return Fibonacci(n - 1) + Fibonacci(n - 2);
    }
}
```

## JavaScript

```execute-javascript
function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

const n = 10;
console.log(fibonacci(n));
```

## Bash

```execute-bash

fibonacci() {
    local n=$1
    if [ $n -le 1 ]; then
        echo $n
    else
        # Recursive calculation of Fibonacci
        local a=$(fibonacci $((n - 1)))
        local b=$(fibonacci $((n - 2)))
        echo $((a + b))
    fi
}

n=10

result=$(fibonacci $n)
echo "$result"
```

## Ruby

```execute-ruby
def fibonacci(n)
    if n <= 1
        n
    else
        fibonacci(n - 1) + fibonacci(n - 2)
    end
end

n = 10
puts fibonacci(n)
```

## Swift

```execute-swift
func fibonacci(_ n: Int) -> Int {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

let n = 10
print(fibonacci(n))
```

## Go

```execute-go
package main

import "fmt"

func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

func main() {
    n := 10
    fmt.Println(fibonacci(n))
}
```

## Rust

```execute-rust
fn fibonacci(n: i32) -> i32 {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

fn main() {
    let n = 10;
    println!("{}", fibonacci(n));
}
```

## Lua

```execute-lua
function fibonacci(n)
    if n <= 1 then
        return n
    else
        return fibonacci(n - 1) + fibonacci(n - 2)
    end
end

local n = 10
print(fibonacci(n))
```

## Haskell

```execute-haskell
fibonacci :: Int -> Int
fibonacci n
    | n <= 1 = n
    | otherwise = fibonacci (n - 1) + fibonacci (n - 2)

main :: IO ()
main = do
    let n = 10
    print $ fibonacci n
```
