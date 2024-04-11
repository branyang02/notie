# **Computer Systems and Organization: Part 2**

<span class="subtitle">
Date: 5/1/2024 | Author: Brandon Yang
</span>

<details><summary>Table of Content</summary>

- [Introduction](#introduction)
- [Building](#building)
  - [Compilation](#compilation)
  - [Static Libraries](#static-libraries)
  - [Dynamic Libraries](#dynamic-libraries)
  - [Makefile](#makefile)
    - [Rules](#rules)

</details>

#### **Introduction**

This is a note for the course Computer Systems and Organization: Part 2 (CSO2) at the University of Virginia.

Example C code:

```c
#include <stdio.h>

int main() {
    printf("Hello, World!\n");
    return 0;
}
```

#### **Building**

##### **Compilation**

- `clang` / `gcc` flags:
  - compile only: `clang -S foo.c` (output: `foo.s`)
  - assemble only: `clang -c foo.s` (output: `foo.o`)
  - **compile and assemble**: `clang -c foo.c` (output: `foo.o`)
  - link only: `clang foo.o bar.o` (output: `a.out`)
  - compile, assemble, and link: `clang foo.c bar.c` (output: `a.out`)
  - **compile, assemble, and link**: `clang foo.c bar.c -o myprog` (output: `myprog`)

##### **Static Libraries**

- Create a static library `libfoo.a`: `ar rcs libfoo.a foo.o bar.o`
- Link with a static library: `cc -o myprog foo.c bar.c -L/path/to/lib -lfoo`

##### **Dynamic Libraries**

- Create a shared library:
  1. Compile with `-fPIC`: `cc -c -fPIC foo.c bar.c` (output: `foo.o`, `bar.o`)
  2. Link with `-shared` to create `libfoo.so`: `cc -shared -o libfoo.so foo.o bar.o`
- Link with a shared library: `cc -o myprog foo.c bar.c -L/path/to/lib -lfoo`

##### **Makefile**

```makefile
target: dependencies
    command
```

- make runs `command` if `target` is older than any of the `dependencies`.

```makefile
# Makefile
CC = clang
CFLAGS = -Wall -Wextra -Werror
LDFLAGS = -L/path/to/lib -lfoo

myprog: main.o libfoo.a
    $(CC) -o myprog main.o $(LDFLAGS)

main.o: main.c
    $(CC) $(CFLAGS) -c main.c

libfoo.a: foo.o bar.o
    ar rcs libfoo.a foo.o bar.o

foo.o: foo.c
    $(CC) $(CFLAGS) -c foo.c

bar.o: bar.c
    $(CC) $(CFLAGS) -c bar.c

clean:
    rm -f myprog main.o libfoo.a foo.o bar.o

.PHONY: clean
```

- Macros: `CC`, `CFLAGS`, `LDFLAGS`
- `PHONY` target: `clean` (not a file)

<details><summary>Practice</summary>

```makefile
# Makefile
W: X, Y
    buildW
X: Q
    buildX
Y: X, Z
    buildY
```

To make sure `W` is up to date, we need to:

- Make sure `X` is up to date.
  - Make sure `Q` is up to date.
- Make sure `Y` is up to date.
  - Make sure `X` is up to date.
    - Make sure `Q` is up to date.
  - Make sure `Z` is up to date.

In summary, Makefile follows the dependency graph to ensure all dependencies are up to date.

</details>

###### **Rules**:

```makefile
# Makefile
CC = gcc
CFLAGS = -Wall
LDFLAGS = -Wall
LDLIBS = -lm

program: main.o extra.o
    $(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
extra.o: extra.c extra.h
    $(CC) $(CFLAGS) -o $@ -c $<
main.o: main.c main.h extra.h
    $(CC) $(CFLAGS) -o $@ -c $<
```

- `$@`: target
- `$^`: all dependencies
- `$<`: first dependency

To build any file ending in `.o`, make should look for a `.c` file with the same stem (the part before the extension) and use the command specified in the rule to compile the `.c` file into an `.o` file.

```makefile
%.o: %.c
$(CC) $(CFLAGS) -o $@ -c $<

```

- `%`: wildcard

#### **Permissions**
