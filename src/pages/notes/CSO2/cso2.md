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
- [Permissions](#permissions)
  - [User IDs](#user-ids)
  - [Group IDs](#group-ids)
  - [File Permissions](#file-permissions)
  - [Changing Permissions](#changing-permissions)
  - [Superuser](#superuser)
- [Kernel + System Calls](#kernel--system-calls)
  - [Kernel Mode vs. User Mode](#kernel-mode-vs-user-mode)
  - [Implementation](#implementation)
  - [Exceptions](#exceptions)
  - [Handling Exceptions](#handling-exceptions)
- [References](#references)
</details>

#### **Introduction**

This is a note for the course Computer Systems and Organization: Part 2 (CSO2) at the University of Virginia. This note contains live code examples and explanations for various topics in the course.

Example _**live**_, _**runnable**_ C code:

```execute-c
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    pid_t pid;

    // Create a new process
    pid = fork();

    if (pid == -1) {
        // If fork() returns -1, an error occurred
        perror("Failed to fork");
        return 1;
    } else if (pid > 0) {
        // Parent process
        printf("I am the parent process. PID: %d, Child PID: %d\n", getpid(), pid);
        // Optionally, wait for the child to exit
        wait(NULL);
    } else {
        // Child process
        printf("I am the child process. PID: %d, Parent PID: %d\n", getpid(), getppid());
        // Execute some code as the child
    }

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

- **Become part of executable (archive of .o files).**
- Create a static library `libfoo.a`: `ar rcs libfoo.a foo.o bar.o`
- Link with a static library: `cc -o myprog foo.c bar.c -L/path/to/lib -lfoo`

##### **Dynamic Libraries**

- **Loaded when executable starts.**
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

###### **Built-in rules**:

`make` has the "make `.o` from `.c`" rule built-in already, so:

```makefile
CC = gcc
CFLAGS = -Wall
LDFLAGS = -Wall
LDLIBS = -lm

program: main.o extra.o
    $(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
```

The built-in rule for compiling `.c` files to `.o` files in make looks something like this:

```makefile
%.o: %.c
    $(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<
```

You can also supply header files as dependencies to ensure that the `.o` files are rebuilt when the headers change:

```makefile
CC = gcc
CFLAGS = -Wall
LDFLAGS = -Wall
LDLIBS = -lm

program: main.o extra.o
    $(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

main.o: main.c main.h extra.h
extra.o: extra.c extra.h
```

#### **Permissions**

##### **User IDs**

- **User ID (UID)**: unique identifier for a user.
- Every process has a user ID.
- User ID used to decide what process is authorized to do.

##### **Group IDs**

- **Group ID (GID)**: unique identifier for a group.

##### **File Permissions**

- 2 types
  - Access control list (ACL): list of permissions attached to an object.
  - `chmod` style permissions.
- Each file has the following permissions:
  - User permissions
  - Group permissions
  - Other permissions
- Each permission can be one of the following:
  - **Read (r)**: read the contents of the file.
  - **Write (w)**: modify the contents of the file.
  - **Execute (x)**: execute the file as a program. (For directories, search the directory.)

###### **Permissions Encoding**

- separated into 3 groups of 3 bits each.
  - user, group, other
- Example:
  - User can read, write, and execute. Group can read and execute Other can read.
    - **Symbolic notation**: `rwxr-xr--`
    - **Octal notation**: `754`
    - **Binary notation**: `111101100`

##### **Changing Permissions**

- **Symbolic notation**:
  - `chmod u+x file`: add execute permission for user.
  - `chmod g-w file`: remove write permission for group.
  - `chmod o=r file`: set read permission for other.
  - `chmod a=rwx file`: set read, write, and execute permission for all.
  - `chmod u=rw,go=r file`: set read and write permission for user, read permission for group and other.
- **Octal notation**:
  - `chmod 754 file`: set permissions to `rwxr-xr--`.

###### **POSIX ACL Syntax**

- **Symbolic notation**:
  - `getfacl file`: get ACL for file.
  - `setfacl -m u:brandon:rw file`: set read and write permission for user `brandon`.
  - `setfacl -m g:staff:r file`: set read permission for group `staff`.
  - `setfacl -m o::r file`: set read permission for other.
  - `setfacl -m u::rwx,g::r-x,o::r-- file`: set read, write, and execute permission for user, read and execute permission for group, read permission for other.
- **Octal notation**:
  - `setfacl -m u::7 file`: set read, write, and execute permission for user.
  - `setfacl -m g::5 file`: set read and execute permission for group.
  - `setfacl -m o::4 file`: set read permission for other.

###### **Superuser**

- **Superuser**: user with special privileges (user ID = 0).
- **Root**: superuser on Unix-like systems.
- **sudo**: run a command as the superuser.

#### **Kernel + System Calls**

##### **Kernel Mode vs. User Mode**

- **Kernel mode**: unrestricted access to hardware.
- **User mode**: restricted access to hardware.
- **Kernel**: the space where the operating system runs.

<div style="display: flex; justify-content: center; align-items: center;">
    <div style="background-color: white;">
        <img src="https://www.cs.virginia.edu/~cr4bd/3130/S2024/readings/kernel-layers1.svg" alt="Kernel Layers" style="display: block; max-height: 100%; max-width: 100%;">
    </div>
</div>
<span
    class="caption">We can view the combination of the limited user-mode hardware interface and system calls as collectively defining the interface user mode code sees.
</span>

##### **Implementation**

- **Mode bit**: bit in the processor that determines the mode. (0 = kernel mode, 1 = user mode)
- **Mode Switch**: change from user mode to kernel mode using **exceptions**.

##### **Exceptions**

- **Interrupts**:
  - occurs independently from the code being executed when it occurs.
  - ex. mouse click while a unrelated program is running
  - handled by the kernel, invisible to the user program
- **Faults**:
  - An instruction failing to suceed in its execution.
  - ex. divide by zero
  - Responses
    - Kernel fixes the problem and restarts the instruction
    - Kernel cannot fix the problem, kill the process
- **Traps**:
  - An exception caused by a special instruction whose purpose is to cause a trap.

##### **Handling Exceptions**

The basic mechanism for any exception to be handled is

1. The processor saves the current state of the program.
2. Enters kernel mode.
3. Jump to code designed to react to the exception in question, called an **exception handler**.
4. When the handler finishes, enter user mode and restore processor state (program counter, kernel mode bit, etc.)

- **Exception Handler**
  - **Exception Table/Vector**: a table of pointers to exception handlers.
  - **Exception Number**: index into the table.

![](https://branyang02.github.io/images/exception_table.png)
<span
    class="caption"> The exception table is a table of pointers to exception handlers. The exception number is used as an index into the table to find the appropriate handler.
</span>

### **References**

This note is based on “[CS 3130 Spring 2024](https://www.cs.virginia.edu/~cr4bd/3130/S2024/)” by Charles Reiss, used under CC BY-NC-SA 4.0.
