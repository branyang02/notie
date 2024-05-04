# **Computer Systems and Organization: Part 2**

<span class="subtitle">
Date: 5/1/2024 | Author: Brandon Yang
</span>

## **Introduction**

These are my notes for Computer Systems and Organization 2 (CSO2) at the University of Virginia in the Spring 2024 semester taught by Charles Reiss. This note contains live code examples and explanations for various topics in the course.

<!-- ```tikz
\begin{tikzpicture}
  \def \n {5}
  \def \radius {3cm}
  \def \margin {8} % margin in angles, depends on the radius

  \foreach \s in {1,...,\n}
  {
    \node[draw, circle] at ({360/\n * (\s - 1)}:\radius) {$\s$};
    \draw[->, >=latex] ({360/\n * (\s - 1)+\margin}:\radius)
      arc ({360/\n * (\s - 1)+\margin}:{360/\n * (\s)-\margin}:\radius);
  }
\end{tikzpicture}
``` -->

## **Building**

### **Compilation**

- `clang` / `gcc` flags:
  - compile only: `clang -S foo.c` (output: `foo.s`)
  - assemble only: `clang -c foo.s` (output: `foo.o`)
  - **compile and assemble**: `clang -c foo.c` (output: `foo.o`)
  - link only: `clang foo.o bar.o` (output: `a.out`)
  - compile, assemble, and link: `clang foo.c bar.c` (output: `a.out`)
  - **compile, assemble, and link**: `clang foo.c bar.c -o myprog` (output: `myprog`)

### **Static Libraries**

- **Become part of executable (archive of .o files).**
- Create a static library `libfoo.a`: `ar rcs libfoo.a foo.o bar.o`
- Link with a static library: `cc -o myprog foo.c bar.c -L/path/to/lib -lfoo`

### **Dynamic Libraries**

- **Loaded when executable starts.**
- Create a shared library:
  1. Compile with `-fPIC`: `cc -c -fPIC foo.c bar.c` (output: `foo.o`, `bar.o`)
  2. Link with `-shared` to create `libfoo.so`: `cc -shared -o libfoo.so foo.o bar.o`
- Link with a shared library: `cc -o myprog foo.c bar.c -L/path/to/lib -lfoo`

### **Makefile**

```makefile
target: dependencies
    command
```

- make runs `command` if `target` is older than any of the `dependencies`.

```makefile
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

#### **Rules**

```makefile
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

#### **Built-in rules**

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

## **Permissions**

### **User IDs**

- **User ID (UID)**: unique identifier for a user.
- Every process has a user ID.
- User ID used to decide what process is authorized to do.

### **Group IDs**

- **Group ID (GID)**: unique identifier for a group.

### **File Permissions**

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

#### **Permissions Encoding**

- separated into 3 groups of 3 bits each.
  - user, group, other
- Example:
  - User can read, write, and execute. Group can read and execute Other can read.
    - **Symbolic notation**: `rwxr-xr--`
    - **Octal notation**: `754`
    - **Binary notation**: `111101100`

### **Changing Permissions**

- **Symbolic notation**:
  - `chmod u+x file`: add execute permission for user.
  - `chmod g-w file`: remove write permission for group.
  - `chmod o=r file`: set read permission for other.
  - `chmod a=rwx file`: set read, write, and execute permission for all.
  - `chmod u=rw,go=r file`: set read and write permission for user, read permission for group and other.
- **Octal notation**:
  - `chmod 754 file`: set permissions to `rwxr-xr--`.

#### **POSIX ACL Syntax**

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

#### **Superuser**

- **Superuser**: user with special privileges (user ID = 0).
- **Root**: superuser on Unix-like systems.
- **sudo**: run a command as the superuser.

## **Kernel + System Calls**

### **Kernel Mode vs. User Mode**

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

### **Implementation**

- **Mode bit**: bit in the processor that determines the mode. (0 = kernel mode, 1 = user mode)
- **Mode Switch**: change from user mode to kernel mode using **exceptions**.

### **Exceptions**

| Exceptions     | Classify by Cause                                                 | Classify by Result                                         |
| -------------- | ----------------------------------------------------------------- | ---------------------------------------------------------- |
| **Interrupts** | occurs independently from the code being executed when it occurs. | runs each instruction once (has no triggering instruction) |
| **Faults**     | An instruction failing to suceed in its execution.                | re-runs triggering instruction                             |
| **Traps**      | caused by a special instruction whose purpose is to cause a trap  | runs instruction after triggering instruction              |

### **Handling Exceptions**

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

### **System Calls**

- **System Call**: a way of communication from user mode to kernel mode.
  - Implemented as a `trap` with exception number `128`. The "action number" is passed into register `%rax`.

<details><summary>Example Socket System Call</summary>

Below is an example of C library function `socket` that makes a system call to create a socket.

```assembly
socket:
    endbr64
    mov    $0x29,%eax
    syscall
```

- `endbr64`: control-flow enforcement. Not relevant to the system call.
- `mov $0x29,%eax`: move `41` (`0x29`) into `%rax`. `41` is the system call number for `socket`.
- `syscall`: A `trap` instruction, generating _**exception number**_ `128`. Then the following happens:
  1. Processor saves the current state of the program.
  2. Enters kernel mode.
  3. Jump to `exception_handler[128]`.
     1. `system_call_handler[41]` is called with `%rax` set to `41`.
  4. When the handler finishes, enter user mode and restore processor state.

</details>

## **Multitasking**

- **Multitasking**: a generic term for having multiple processes running on a single machine.
- **Preemptive Multitasking**: the operating system can interrupt a process and switch to another process.
- **Cooperative Multitasking**: the process must voluntarily give up control.

### **Processes**

- **Process**: an instance of a program in execution, acts like a _virtual machine_.
  - A process has its own program registers, condition codes, **virtual address space**, etc.
- **Virtual Address Space**: the memory that a process can access. (illusion of a program having its own memory)

<img src="https://branyang02.github.io/images/address_space.png" alt="Virtual Address Space" style="display: block; max-height: 70%; max-width: 70%;">
<span
    class="caption"> The virtual address space is the memory that a process can access. It is an illusion of a program having its own memory.
</span>

#### **Context Switch**

- **Context Switch**: the process of saving the state of a process and loading the state of another process.
  1. OS starts running a process.
  2. Exception occurs.
  3. OS saves the state of the current process (old registers, program counter, mapping of addresses(**page tables**), etc).
  4. OS loads the state of another process.
  5. OS starts running the new process.

| Program A Running (Before)                                    | Program B Running (After)                                     |
| ------------------------------------------------------------- | ------------------------------------------------------------- |
| ![](https://branyang02.github.io/images/context_switch_A.png) | ![](https://branyang02.github.io/images/context_switch_B.png) |

<details><summary>Time Multiplexing</summary>

Linux uses time multiplexing to switch between processes, which refers to "sharing the processor over time". The kernel uses a **timer** to interrupt the current process and switch to another process using a **context switch**.

Suppose we have two processes, `A` and `B`, and a timer interrupt every `10ms`. Here is a timeline of the processes:

1. `A` starts running.
2. After `10ms`, timer expires, triggering an `interrupt` exception.
3. Enter kernel mode.
4. Save the state of `A`.
5. Load the state of `B`.
6. Return to user mode and start running `B`.

</details>

#### **Process** vs. **Thread**

- **Process**: an instance of a program in execution.
- **Thread**: a process can have multiple threads of execution. Threads share the same **virtual address space**, but have their own **program registers**, **program counter**, condition codes, etc.

  <div style="background-color: white;">
      <img src="https://static.javatpoint.com/difference/images/process-vs-thread3.png" style="max-height: 70%; max-width: 70%;">
  </div>
  <span
      class="caption"> Threads within the same process share the same virtual address space but have their own program registers, program counter, condition codes, etc. (Source: javapoint, <a href="https://www.javatpoint.com/process-vs-thread">Process Vs. Thread</a>)
  </span>

## **Signals**

- **Signal**: a way to notify a process that an event has occurred.
- **Signal Handler**: a function that is called when a signal is received.
  - Ex. `SIGINT` (interrupt from keyboard), `SIGSEGV` (segmentation fault), `SIGKILL` (kill the process).

### **Signal vs. Exception**

|                 | User code     | Kernel code   | Hardware               |
| --------------- | ------------- | ------------- | ---------------------- |
| **User code**   | ordinary code | Trap          | via kernel             |
| **Kernel code** | **Signal**    | ordinary code | protected instructions |
| **Hardware**    | via kernel    | Interrupt     | —                      |

<span
    class="caption"> Signals are roughly the kernel-to-user equivalent of an interrupt. At any time, while executing any line of code, a signal may appear.
</span>

|                        | (hardware) exceptions              | signals                         |
| ---------------------- | ---------------------------------- | ------------------------------- |
| **Handler Mode**       | handler runs in **kernel mode**    | handler runs in **user mode**   |
| **Decision Maker**     | hardware decides when              | OS decides when                 |
| **State Saving**       | hardware needs to save PC          | OS needs to save PC + registers |
| **Instruction Change** | processor next instruction changes | thread next instruction changes |

<span
    class="caption"> Signals vs. Exceptions
</span>

### **Forwarding exceptions to signals**

<img src="https://branyang02.github.io/images/signals.png" alt="Signals" style="display: block; max-height: 70%; max-width: 70%;">
<span
      class="caption"> When `SIGINT` is received, the program enters kernel mode and starts running the exception handler for handing keyboard interrupts. The exception handler then forwards the signal to the user mode signal handler. The signal handler then runs in user mode. After the signal handler finishes, the program enters the kernel mode again to clean up and return to user mode.
</span>

### **Common Signals**

| Constant             | Likely Use                                                   |
| -------------------- | ------------------------------------------------------------ |
| `SIGBUS`             | "bus error"; certain types of invalid memory accesses        |
| `SIGSEGV`            | "segmentation fault"; other types of invalid memory accesses |
| `SIGINT`             | what control-C usually does                                  |
| `SIGFPE`             | "floating point exception"; includes integer divide-by-zero  |
| `SIGHUP`, `SIGPIPE`  | reading from/writing to disconnected terminal/socket         |
| `SIGUSR1`, `SIGUSR2` | use for whatever you (app developer) wants                   |
| `SIGKILL`            | terminates process (**cannot be handled by process!**)       |
| `SIGSTOP`            | suspends process (**cannot be handled by process!**)         |

### **Signals Setup**

- **Signal API**
  - `sigaction()`: set up a signal handler.
  - `raise(sig)`: send a signal to the _current_ process.
  - `kill(pid, sig)`: send a signal to a process with a specific PID.
    - Bash: `kill 1234` sends `SIGTERM` to process with PID `1234`.
    - C: `kill(1234, SIGTERM)` sends `SIGTERM` to process with PID `1234`.
    - Bash: `kill -USR1 1234` sends `SIGUSR1` to process with PID `1234`.
    - C: `kill(1234, SIGUSR1)` sends `SIGUSR1` to process with PID `1234`.
  - `SA_RESTART`
    - when included: after signal handler runs, attempt to restart the interrupted operation. (e.g., reading from keyboard)
    - when not included: after signal handler runs, return `-1` with `errno` set to `EINTR`.
- `kill()` not always immediate.
  - Ex. In a multi-core system, the OS records the signal and sends it to the process when it is ready.

```c
#include signal.h

static void handler(int signum) {
    // Handle what to do when signal is received
}

int main() {
    struct sigaction sa;
    sa.sa_handler = &handler;  // Set the handler function
    sigemptyset(&sa.sa_mask); // Initialize the signal set to empty
    sa.sa_flags = SA_RESTART;
    sigaction(SIGINT, &sa, NULL); // Register the signal handler for SIGINT

    // Run normal program code

    return 0;
}
```

<details><summary>Signal Handler Example</summary>

Below is an example of a signal handler that simulates `SIGINT` (interrupt from keyboard).

```execute-c
#define _POSIX_C_SOURCE 200809L

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void handler(int signum) {
    write(1, "SIGINT received. Provide end-of-file to end program.\n",
          strlen("SIGINT received. Provide end-of-file to end program.\n"));
    write(1, "Signal handler reached. Exiting now.\n",
          strlen("Signal handler reached. Exiting now.\n"));
    exit(0);
}

int main() {
    struct sigaction sa;
    sa.sa_handler = &handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;

    if (sigaction(SIGINT, &sa, NULL) == -1) {
        fprintf(stderr, "unable to override SIGINT signal\n");
        return 1;
    }

    // Raise SIGINT signal to trigger the handler
    raise(SIGINT);

    fprintf(stderr, "This should not be printed.\n");

    return 0;
}
```

- `write` instead of `printf` in `handler`
  - `printf` is not async-signal-safe. (not safe to call in a signal handler)
- `void handler(int signum)`
  - Signal handler function. `signum` is the signal number.
- `struct sigaction sa`
  - Ttructure to specify the action to be taken on a signal.
- `sa.sa_handler = &handler`
  - The function pointer to invoke.
- `sigemptyset(&sa.sa_mask)`
  - Initializes the signal set to empty. Do not "block" additional signals while signal handler is running.
- `sa.sa_flags = SA_RESTART`
  - Restart system calls if interrupted by a signal.
- `sigaction(SIGINT, &sa, NULL)`
  - Register the signal handler for `SIGINT`.
- `raise(SIGINT)`
  - Raise the `SIGINT` signal to trigger the handler. (simulate `Ctrl+C`)

</details>

### **Handling multiple signals**

We can use function parameter `signum` to determine which signal was received.

```c
static void handle_signal(int signum) {
    if (signum == SIGINT) {
        write(STDOUT_FILENO, "Caught SIGINT!\n", 15);
    } else if (signum == SIGTERM) {
        write(STDOUT_FILENO, "Caught SIGTERM!\n", 16);
    }
}

int main() {
    struct sigaction sa;
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    // Set up handlers for both SIGINT and SIGTERM
    if (sigaction(SIGINT, &sa, NULL) == -1) {
        perror("Error: cannot handle SIGINT");
    }
    if (sigaction(SIGTERM, &sa, NULL) == -1) {
        perror("Error: cannot handle SIGTERM");
    }
}
```

### **Signal Handler Unsafety**

<blockquote class="important">

Signal handlers should be **async-signal-safe**. This means that the functions called in the signal handler should not interfere with the normal execution of the program.

</blockquote>

- **Async-signal-safe functions**: functions that can be safely called from a signal handler.
  - `write()`, `exit()` ...
  - **DO NOT** use `printf()`, `malloc()`, `free()` ...

We can also avoid running the signal handler while it is already running by blocking the signal.

### **Blocking Signals**

<img src="https://branyang02.github.io/images/block_signal.png" alt="Blocking Signals" style="display: block; max-height: 30%; max-width: 30%;">

We can block signals with `sigprocmask()` to prevent the signal handler from running immediately. When the signal is received, it will be _pending_ until it is unblocked, at which point the signal handler will run.

```c
sigset_t sigint_as_set;
sigemptyset(&sigint_as_set);
sigaddset(&sigint_as_set, SIGINT);
sigprocmask(SIG_BLOCK, &sigint_as_set, NULL);
... /* do stuff without signal handler running yet */
sigprocmask(SIG_UNBLOCK, &sigint_as_set, NULL);
```

- `sigprocmask()` temporarily disables the signal handler from running. If a signal is sent to a process while it is blocked, then the OS will track that is pending. When the pending signal is unblocked, its signal handler will be run.
- `sigsuspend()` temporarily unblocks a blocked signal just long enough to run its signal handler.
- `sigwait()` waits for a signal to be received, blocking until the signal is received. This is used typically _instead of having signal handlers_.

## **Processes**

### **Process Creation**

- `pid_t fork()`: creates a new process by duplicating the calling process.
  - Returns `0` to the child process, returns the **child process's PID** to the parent process.
- `pid_t getpid()`: returns the PID of the calling process.

When we create a new process, the child process is a **copy** of the parent process. The child process has its own virtual address space, but it shares the same code, data, and heap as the parent process.

<details><summary>Example Fork</summary>

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

- `pid_t pid`: variable to store the return value of `fork()`.
- `pid = fork()`: create a new process.

</details>

If we are running in a **single-core** setting, then running the parent and child processes require **context switching**. However, if we are running in a **multi-core** setting, then the parent and child processes can run concurrently.

<blockquote class="important">

The parent and child processes may run concurrently, so the order of output may vary.

</blockquote>

### **Process Management**

- `waitpid()`: wait for a specific child process to exit.

We can use `waitpid()` to wait for a specific child process to **exit**. The function `waitpid()` blocks the parent process until the child process with the specified PID exits.

<blockquote class="important">

`waitpid()` only works for child processes.

</blockquote>

<details><summary>Example Waitpid</summary>

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
        // Wait for the child to exit
        waitpid(pid, NULL, 0);
        printf("I am the parent process. PID: %d, Child PID: %d\n", getpid(), pid);
    } else {
        // Child process
        printf("I am the child process. PID: %d, Parent PID: %d\n", getpid(), getppid());
        // Execute some code as the child
    }

    return 0;
}
```

In this example, the parent process waits for the child process to exit using `waitpid(pid, NULL, 0)`. Therefore, the child process will run and print to console before the parent process prints to console.

- `waitpid(pid, NULL, 0)`: wait for the child process with PID `pid` to exit.

</details>

We can also use `exec()` functions to replace the current process with a new process. For example, we may want to replace the current process with a new shell process:

```c
if (child_pid == 0) {
  // child process
  char *args[] = {"ls", "-l", NULL};
  execv("/bin/ls", args);
  // execv doesn't return unless there is an error
  perror("execv");
  exit(1);
} else {
  // parent process
  // ...
}
```

`exec` will simply run the new program and exit when the new program exits.

### **File Descriptors**

In Unix, every process has an array of _open file descriptions_ that point to open files.

<blockquote class="definition">

A **file descriptor** is a non-negative integer that serves as the index into the array of open file descriptions.

</blockquote>

- **Standard File Descriptors**:
  - `0`: standard input (stdin)
  - `1`: standard output (stdout)
  - `2`: standard error (stderr)

**Getting File Descriptors**

- `int open(const char *pathname, int flags)`: open a file and return a file descriptor.
- `int close(int fd)`: close a file descriptor, returning `0` on success.
  - `close()` simply deallocates the file descriptor, it does not delete the file, and does not affect other file descriptors.

#### **Redirecting File Descriptors**

We can manually **redirect** file descriptors. For example, we can perform shell redirection like this:

- `./my_program ... < input.txt`
  - run `my_program` with `stdin` redirected from `input.txt`.
- `echo foo > output.txt`
  - run `echo` with `stdout` redirected to `output.txt`.

#### **Dup2**

When we `fork` a process, the child process inherits the parent's file descriptors. However, we can **redirect** the child process's file descriptors using `dup2()`.

- `int dup2(int oldfd, int newfd)`: make `newfd` refer to the same open file as `oldfd`.
  - Ex. `dup2(fd, STDOUT_FILENO)`: overrwrites what `STDOUT_FILENO` points to with `fd`.

<details><summary>Dup2 Example</summary>
  
````execute-c
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <stdlib.h>

int main() {
pid_t pid;
int fd;

    // Open a file
    fd = open("output.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);

    // Create a new process
    pid = fork();

    if (pid == -1) {
        // If fork() returns -1, an error occurred
        perror("Failed to fork");
        return 1;
    } else if (pid > 0) {
        // Parent process
        // Wait for the child to exit
        waitpid(pid, NULL, 0);
        printf("I am the parent process. PID: %d, Child PID: %d\n", getpid(), pid);

        printf("This is the content of output.txt:\n");
        fflush(stdout);

        // view output.txt
        char *args[] = {"cat", "output.txt", NULL};
        execv("/bin/cat", args);

        perror("execv");
        exit(1);
    } else {
        // Child process
        // Redirect stdout to the file
        dup2(fd, STDOUT_FILENO);
        printf("I am the child process. PID: %d, Parent PID: %d\n", getpid(), getppid());
        close(fd);  // optionally, we can close the file descriptor
    }
    return 0;

}

````

In this example, the child process redirects `stdout` to the file `output.txt` using `dup2(fd, STDOUT_FILENO)`. Therefore, the child process will write to `output.txt` instead of the console.

</details>


#### **Pipes**

- **Pipe**: a unidirectional communication channel between two processes.
  - One process writes to the pipe, and the other reads from the pipe.
- `pipe(int pipefd[2])`: create a pipe and return two file descriptors.
  - `pipefd[0]`: read end of the pipe.
  - `pipefd[1]`: write end of the pipe.

When writing to the write end of the pipe, we follow these steps:
1. `close(readfd)`: close the read end of the pipe.
2. Write to the write end of the pipe.
3. `close(writefd)`: close the write end of the pipe.

When reading from the read end of the pipe, we follow these steps:
1. `close(writefd)`: close the write end of the pipe.
2. Read from the read end of the pipe.
3. `close(readfd)`: close the read end of the pipe.

<blockquote class="important">

`pipe` must be called before `fork` to ensure that the parent and child processes share the pipe.

</blockquote>

<details><summary>Pipe Example</summary>

```execute-c
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>

int main() {
    int pipe_fd[2];
    if (pipe(pipe_fd) == -1) {
        perror("pipe");
        return 1;
    }
    int readfd = pipe_fd[0];
    int writefd = pipe_fd[1];

    pid_t child_pid = fork();
    if (child_pid == -1) {
        perror("fork");
        return 1;  // Fork failed
    }

    if (child_pid == 0) {
        // Child process, write to pipe
        close(readfd);  // Close the unused read end
        char message[] = "Hello, parent!\n";
        write(writefd, message, strlen(message));  // Write the message to the pipe
        close(writefd);  // Close write end after writing
        return 0;
    } else {
        // Parent process, read from pipe
        close(writefd);  // Close the unused write end
        char buf[100];
        int nbytes = read(readfd, buf, sizeof(buf)-1);  // Read from the pipe
        if (nbytes > 0) {
            buf[nbytes] = '\0';  // Null-terminate the string
            printf("Parent read: %s", buf);
        }
        close(readfd);  // Close read end
    }

    return 0;
}

```

In this example, the following process happens:

1. call `pipe` to create a pipe with two file descriptors.
2. call `fork` to create a child process.
3. in the child process, close the read end of the pipe, write a message to the pipe, and close the write end of the pipe.
4. in the parent process, close the write end of the pipe, read from the pipe, close the read end of the pipe.


The parent process also waits for the child process to finish without using `waitpid`.

</details>



## **Virtual Memory**

Every process has its own **virtual address space** when running a program.

<img src="https://branyang02.github.io/images/address_space.png" alt="Virtual Address Space" style="display: block; max-height: 50%; max-width: 50%;">

<blockquote class="definition">

**Virtual Address Space**: the memory that a process can access. It is an illusion of a program having its own memory.

</blockquote>

We can split any memory address (both virtual and physical) into two parts: the **page number** and the **page offset**.

- **Page Number**: high-order bits of the address.
- **Page Offset**: low-order bits of the address.



To access the **physical memory**, the OS provides a **page table** that maps virtual addresses to physical addresses. The **page number** is used as the key to index into the **Page Table**, and the **page offset** is used to access the data within the page.

### **Page Table**

<blockquote class="definition">

A **page table** is a data structure that maps virtual addresses to physical addresses.

</blockquote>

A simple page table may look like this:

<div class="small-table">

| Valid? | Physical Page Number |
| ------ | -------------------- |
| 1 | 010 |
| 0 | —   |
| 1 | 101 |
| 1 | 110 |

</div>

In the page table above, the index of the row corresponds to the **virtual page number (VPN)**, or the page number of the virtual address. Each row is called a **page table entry (PTE)**, where it contains a **valid bit** to indicate if the translation is valid or not, and the corresponding **physical page number (PPN)**.

Since the page table is simply a _translation_ from virtual addresses to physical addresses, we define the main steps of the translation process:

1. Given the virtual address, extract the **VPN** and **page offset**.
2. Use the **VPN** to index into the **page table** and retrieve the **PTE**.
3. If the **valid bit** is set, then the **PPN** is valid; otherwise, the translation is invalid and an exception is raised.
4. Concatenate the **PPN** with the **page offset** to form the physical address.


<blockquote class="important">

We have a **different** page table for each process.

</blockquote>


### **Address Space Size**

<blockquote class="definition">

The **address space size** is the maximum number of unique addresses that can be generated ba processor.

</blockquote>

<blockquote class='equation'>

$$
\begin{equation*}
\text{Address Space Size} = 2^{\text{number of bits}}
\end{equation*}
$$

</blockquote>

For example, if we want to calculate the **virtual address space size** given that we have 32-bit virtual addresses:

$$
\begin{equation*}
\text{Virtual Address Space Size} = 2^{32} \text{ bytes}.
\end{equation*}
$$

Similarly, if we want to calculate the **physical address space size** given that we have 20-bit physical addresses:

$$
\begin{equation*}
\text{Physical Address Space Size} = 2^{20} \text{ bytes}.
\end{equation*}
$$

This is because each unique address stores **1** byte of data, and the number of unique addresses is equal to the size of the address space.

We can also calculate other things once we know the address space size:


<blockquote class="equation">

$$
\begin{align*}
\text{Number of Virtual Pages} &= \frac{\text{Virtual Address Space Size}}{\text{Page Size}} \\
\text{Number of Physical Pages} &= \frac{\text{Physical Address Space Size}}{\text{Page Size}} \\
\\
\text{VPN bits} &= \log_2(\text{Number of Virtual Pages}) \\
\text{PPN bits} &= \log_2(\text{Number of Physical Pages}) \\
\text{offset bits} &= \log_2(\text{Page Size}) \\
\\
\text{Number of PTEs} &= \text{Number of Virtual Pages} \\
\end{align*}
$$


</blockquote>


<details><summary>Exercise: page counting</summary>

**Q1**: Suppose we have $32$-bit virtual addresses, and each page is $4096$ bytes. Calculate the number of virtual pages.

Answer: Since each virtual address has $32$ bits, we have a total of $2^{32}$ unique virtual addresses, which corresponds to $2^{32}$ bytes of data. Since each page can store $4096$ bytes of data, we can calculate the number of virtual pages as follows:

$$
\begin{equation*}
\text{Number of Virtual Pages} = \frac{2^{32} \text{ bytes}}{4096 \text{ bytes/page}} = \frac{2^{32}}{2^{12}} = 2^{20} \text{ pages}.
\end{equation*}
$$


**Q2**: Suppose we have:
- $32$-bit virtual addresses,
- $30$-bit physical addresses,
- each page is $4096$ (or $2^{12}$) bytes,
- PTE have PPN, and valid bit.

Calculate the size of the page table.

Answer: To find the size of the overall page table, we need to know the number of PTEs, and the size of each PTE. Each PTE is made up of a single valid bit and a PPN. Therefore, we need to calculate the size of each PPN. First, we can find the number of physical pages:

$$
\begin{align*}
\text{Number of Physical Pages} &= \frac{2^{30}}{2^{12}} = 2^{18} \text{ pages}
\end{align*}
$$

Next, we can calculate the PPN bits:

$$
\begin{align*}
\text{PPN bits} &= \log_2(\text{Number of Physical Pages}) = \log_2(2^{18}) = 18 \text{ bits}
\end{align*}
$$

Alternatively, we can calculate the PPN bits by finding the number of **offset** bits. Given that we have $2^{20}$ virtual pages, we can first find the number of VPN bits:

$$
\begin{align*}
\text{VPN bits} &= \log_2(\text{Number of Virtual Pages}) = \log_2(2^{20}) = 20 \text{ bits}
\end{align*}
$$

Since we have $32$-bit virtual addresses, we can calculate the number of **offset** bits:

$$
\begin{align*}
\text{Offset bits} &= 32 - \text{VPN bits} = 32 - 20 = 12 \text{ bits}
\end{align*}
$$

Since both the virtual and physical addresses have the same number of offset bits, we can calculate the number of PPN bits as:

$$
\begin{align*}
\text{PPN bits} &= 32 - \text{Offset bits} = 32 - 12 = 20 \text{ bits}
\end{align*}
$$


Therefore, each PTE is made up of $1$ valid bit and $18$ PPN bits. Next, we need to find the number of PTEs. Since we have already found that we have $2^{20}$ virtual pages, we have $2^{20}$ PTEs. Therefore, the size of the page table is:

$$
\begin{align*}
\text{Size of Page Table} &= \text{Number of PTEs} \times \text{Size of each PTE} \\
&= 2^{20} \times (1 + 18) \text{ bits} \\
&= 2^{20} \times 19 \text{ bits} \\
\end{align*}
$$


</details>


### **Permission Bits**

Usually, a page table entry contains permission bits that specify the access rights for the page. For example, the permission bits may include:

- **Valid**: indicates if the translation is valid.
- **User**: allows user-level code to access the page.
- **Write**: allow writing to the page.
- **Execute**: allow executing code on the page.

<div class="small-table">

| Valid | User | Write | Execute | Physical Page Number |
| ----- | ---- | ----- | ------- | -------------------- |
| 1     | 1    | 1     | 0       | 010                  |
| 0     | 0    | 0     | 0       | —                    |
| 1     | 1    | 0     | 1       | 101                  |


</div>

Permission bits allow page tables to enforce memory protection. For example, if a process tries to write to a read-only page, the OS can raise an exception.

### **Space On Demand**

<blockquote class="definition">

**Space on Demand** is a technique where the OS only loads pages into memory when they are needed.


</blockquote>

When a process is created, the OS does not load the entire program into memory. Instead, the OS only loads the pages that are needed. When a process tries to access a page that is not in memory, the OS raises a **page fault**.

<blockquote class="important">

A **page fault** is an exception raised by the hardware when a process tries to access a page that is not in memory (valid bit is 0).

</blockquote>

When a page fault occurs, the OS dynamically loads the page from disk into memory. The OS then updates the page table entry to indicate that the page is now in memory. This is known as **allocate on demand**.

<details><summary>Allocate Stack Space on Demand Example</summary>

Suppose we have the following initial page table:

<img src="https://branyang02.github.io/images/allocate_on_demand.png" alt="Page Table" style="display: block; max-height: 30%; max-width: 30%;">

We have the following code that needs to be executed

```assembly
pushq %rbx

movq 8(%rcx), %rbx
addq %rbx, %rax
```

Suppose the next instruction `pushq %rbx` referes to the page table at VPN `0x7FFFB`. The following occurs:

1. OS sees that valid bit is `0`, triggers a **page fault**.
2. OS looks up what happend, and realized that the program wants more stack space.
3. OS allocates the new stack space _on demand_.
4. OS returns to the program, and reruns `pushq %rbx`.


</details>

### **Copy on Write**

<blockquote class="definition">

**Copy on Write** is a technique where the OS creates a copy of a page only when a write operation occurs.

</blockquote>

When a new process is created using `fork()`, the child process creates its own page table that points to the same physical pages as the parent process. The OS uses **copy on write** to avoid creating a new copy of the physical pages until a write operation occurs.

<details><summary>Copy on Write Example</summary>

For example, suppose we have the following page table for a process

<div class="xsmall-table">

| Valid? | Write? | PPN |
| ------ | ------ | -------------------- |
|1|1|`0x12345`|
|1|1|`0x12347`|
|1|1|`0x12340`|
|1|1|`0x200DF`|
|1|1|`0x200AF`|

</div>

When we call `fork()`, the child process creates a new page table that points to the same PPN, but with the **write bit** set to `0`. The parent process also sets the **write bit** to `0` in its page table.

<div class="small-table">


<table>
<tr><th>Parent Process Page Table </th><th>Child Process Page Table</th></tr>
<tr><td>


| Valid? | Write? | PPN |
| ------ | ------ | -------------------- |
|1|0|`0x12345`|
|1|0|`0x12347`|
|1|0|`0x12340`|
|1|0|`0x200DF`|
|1|0|`0x200AF`|


</td><td>


| Valid? | Write? | PPN |
| ------ | ------ | -------------------- |
|1|0|`0x12345`|
|1|0|`0x12347`|
|1|0|`0x12340`|
|1|0|`0x200DF`|
|1|0|`0x200AF`|


</td></tr> </table>

</div>

When a write operation occurs in the child process, the OS triggers a **page fault**. The OS then creates a new copy of the physical page and updates the child process's page table to point to the new physical page. Suppose the child process attempts to write to the page at row 5:

<div class="small-table">


<table>
<tr><th>Parent Process Page Table </th><th>Child Process Page Table</th></tr>
<tr><td>


| Valid? | Write? | PPN |
| ------ | ------ | -------------------- |
|1|0|`0x12345`|
|1|0|`0x12347`|
|1|0|`0x12340`|
|1|0|`0x200DF`|
|1|0|`0x200AF`|


</td><td>


| Valid? | Write? | PPN |
| ------ | ------ | -------------------- |
|1|0|`0x12345`|
|1|0|`0x12347`|
|1|0|`0x12340`|
|1|0|`0x200DF`|
|$\color{red}1$|$\color{red}1$| $\color{red}\text{0x300FD}$ |


</td></tr> </table>

</div>

After allocating a copy, the OS reruns the write operation in the child process.

</details>

### **Page Table Storage**

#### **In Memory**

One of the options is to store the page table as an array in memory. To do this, we need to have a base register that points to the start of the page table.

<blockquote class="definition">

A **page table base register** is a register that holds the starting address of the page table. It also corresponds to index `0` of the page table.

</blockquote>

The OS has predefinied which bits at each memory address are used for the permission bits, PPN, and other unused bits. For example, a possible page table layout may look like this:

<div class="small-table">

| valid (bit 15)  | PPN (bits 4 - 14)  | unused (bits 0 - 3) |
| --------------- | ------------------- | ------------------- |
|... | ... | ... |

</div>

Therefore, to interpret PTE at a given address, we need to extract the bits that correspond to the valid bit and PPN. The overall process can be described in the following diagram:

<img src="https://branyang02.github.io/images/page_table.png" alt="Page Table Memory" style="display: block; max-height: 70%; max-width: 70%;">

We can also now formally write a formula that describes the process of translating a virtual address to a physical address:

<blockquote class="equation">

Given a virtual address $\text{VA}$, the **Page Table Base Register** $\text{PTBR}$, and the **Page Table Entry Size** $\text{PTE size}$, we can calculate the physical address $\text{PA}$ as:

$$
\begin{align*}
\text{VPN} &= \text{VA} \text{ >> offset bits} \\
\text{PTE} &= M \left[\text{PTBR} + \text{VPN} \times \text{PTE size}\right] \\
\text{PPN} &= \text{PTE} \text{ \& }\text{PPN mask} \\
\text{PA} &= \text{PPN} \text{ << offset bits} + \text{offset}
\end{align*}
$$

where $M$ is the memory array that stores the page table, and $\text{PPN mask}$ is a mask that extracts the PPN bits from the PTE.

</blockquote>

<blockquote class="important">

If you want the actual **value** that is stored at a physical address, make sure to index into the memory array using the **physical address**.

$$
\text{Value} = M[\text{PA}].
$$

</blockquote>

#### **Multi-Level Page Table**

We can also store the page table as a **multi-level page table** in a _tree-like data structure_. This is useful when the page table is too large to fit in memory.

In a multi-level page table, the virtual address is split into multiple parts, each part corresponding to a level in the page table. The OS uses the first part of the virtual address to index into the first level of the page table, and the second part to index into the second level, and so on.

<blockquote class="equation">

We split the VPN into $n$ parts, where $n$ is the number of levels in the page table:

$$
\text{VPN} = \text{VPN}_1, \text{VPN}_2, \ldots, \text{VPN}_n
$$


</blockquote>

Each level of the page table contains a regular PTE and PPN, but the PPN is converted to a physical address to perform the next level of indexing. The final page table entry contains the _actual_ PPN, which is used to access the physical memory.

**Converting PPN to Physical Address for next level lookup**

Once we retrieve the PPN in a page level lookup, we extract the PPN from the corresponding PTE, and **concatenate** it with the number of offset bits of `0`'s to find the _base address_ of the next level of the page table.

<blockquote class="important">

When checking for permission bits, if a PTE is invalid, the OS does **NOT** need to check the next level of the page table.

</blockquote>

<details open><summary>Multi-Level Page Table Example</summary>

Suppose we have the following memory layout:

<img src="https://branyang02.github.io/images/pt_example.png" alt="Multi-Level Page Table" style="display: block; max-height: 50%; max-width: 50%;">

and the following configuration:

- 9-bit virtual addresses
- 6-bit physical addresses
- 8 byte pages
- 1 byte PTE size
- PTE contains: 3 bit PPN, 1 valid bit, 4 unused bit.
- page table base register at `0x20`
- 2-level page table

We want to translate the virtual address `0x129`.

1. We find the VPN and offset bits:

$$
\begin{align*}
  \text{VA space} = 2^9 &= 512 \text{ bytes} \\
  \text{\# of PTEs} = \frac{512}{8} &= 64 \text{ entries}
\end{align*}
$$

64 entries correspond to 6 bits of VPN, and 3 bits of offset. Therefore, we extract the VPN and offset bits from the lower 9 bits of the virtual address:

$$
\text{0x}129 = 0001 \; 0010\; 1001 =  \underbrace{1 \; 0010 \; 1}_{\text{VPN bits}} \underbrace{001}_{\text{offset bits}}
$$

Since we are dealing with 2-level page table, we split the VPN bits into two parts evenly:

$$
\text{VPN}_1 = 100_2, \quad \text{VPN}_2 = 101_2.
$$

Now, we can start the first level translation process. We first index into the first level of the page table to find the corresponding PTE:

$$
\begin{align*}
  \text{PTE}_1 &= M \left[\text{PTBR} + \text{VPN}_1 \times \text{PTE size}\right] \\
  &= M\left[\text{0x}20 + 100_2 \times 1 \text{ byte}\right] \\
  &= M\left[\text{0x}24\right] \\
  &= \text{0x}F4 = 1111 \; 0100_2
\end{align*}
$$

Based on the configuration, we can splite the PTE into the valid bit and PPN:

$$
\begin{align*}
  \text{PTE}_1 = \underbrace{111}_{\text{PPN}}\underbrace{1}_{\text{valid bit}} \; 0100_2
\end{align*}
$$

We then find the next level page table's base address by concatenating the PPN with the offset bits of `0`'s:

$$
\begin{align*}
  \text{Base Address} &= \text{PPN} \text{ << offset bits} \\
  &= 111_2 \text{ << } 3 \\
  &= 111000_2
\end{align*}
$$

We then perform the second level translation process. We index into the second level of the page table to find the corresponding PTE:

$$
\begin{align*}
  \text{PTE}_2 &= M \left[\text{Base Address} + \text{VPN}_2 \times \text{PTE size}\right] \\
  &= M\left[\text{0x}38 + 101_2 \times 1 \text{ byte}\right] \\
  &= M\left[\text{0x}3D\right] \\
  &= \text{0x}DC = 1101 \; 1100_2
\end{align*}
$$

Based on the configuration, we can splite the PTE into the valid bit and PPN:

$$
\begin{align*}
  \text{PTE}_2 = \underbrace{110}_{\text{PPN}}\underbrace{1}_{\text{valid bit}} \; 1100_2
\end{align*}
$$

Finally, we find the physical address by concatenating the PPN with the offset bits:

$$
\begin{align*}
  \text{PA} &= \text{PPN} \text{ << offset bits} + \text{offset} \\
  &= 110_2 \text{ << } 3 + 001_2 \\
  &= 110000_2 + 001_2 \\
  &= 110001_2 = \text{0x}31
\end{align*}
$$

If we want to find the value that corresponds to the physical address `0x31`, we can index into the physical memory array at that address to find the value.

$$
\begin{align*}
  M[\text{0x}31] &= \text{0x}0A.
\end{align*}
$$

</details>


## **Cache**

Make things go 🚀🚀🚀

### **Memory Hierarchy**

The memory hierarchy is a structure that organizes memory into different levels based on their access speed and size. The memory hierarchy typically consists of:

- **Registers**: the fastest and smallest memory.
- **L1 Cache**: a small, fast cache that is close to the CPU.
- **L2 Cache**: a larger cache that is slower than L1 cache.
- **Main Memory (RAM)**: slower than cache but larger in size.
- **Secondary Storage (HDD/SSD)**: the slowest and largest storage.

Cache can also be shared among multiple cores. One typical configuration can be seen below:

<img src="https://branyang02.github.io/images/cache_sharing.png" alt="Memory Hierarchy" style="display: block; max-height: 50%; max-width: 50%;">

<span class="caption">
  Each core has its own L1 cache (instruction and data), and its own L2 cache. The L3 cache is shared among all cores.
</span>

#### **Assumptions**

We assume **temporal** and **spatial** locality when designing the memory hierarchy.

<blockquote class="definition">

- **Temporal Locality**: the principle that programs access the same memory locations frequently.
- **Spatial Locality**: the principle that programs access memory locations near each other.

</blockquote>

### **Cache Block**

A **cache block** stores a block of data from the main memory. The size of the cache block is determined by the **block size**. The block size is typically a power of 2, such as 32 bytes or 64 bytes.

This is an example of a cache block that stores 2 bytes of data:

<img src="https://branyang02.github.io/images/cache_block.png" alt="Cache Block" style="display: block; max-height: 50%; max-width: 50%;">

Multiple cache blocks form the **cache table**. The cache table is essentially a lookup table to quickly retrieve data stored in the cache blocks, if it exists.


### **Direct-Mapped Cache**

A **direct-mapped cache** is a type of cache where each block of main memory maps to exactly one block in the cache. Each row in the table is called a **set**, and each set contains bookeeping information such as the **tag**, **valid bit**, as well as a **cache block** that stores the data from the main memory.

<blockquote class="definition">

**Tag**: a unique identifier that corresponds to a block of data in the main memory.

</blockquote>

<blockquote class="definition">

**Set**: a row in the cache that contains a cache block.

</blockquote>

Below is an example of a direct-mapped cache:

<img src="https://branyang02.github.io/images/direct_mapped_cache.png" alt="Direct-Mapped Cache" style="display: block; max-height: 30%; max-width: 30%;">

In this example, the cache has 8 sets, and each set contains a valid bit, a tag, and a cache block. The cache block stores 2 bytes of data from the main memory.

### **Tag-Index-Offset**

To access a block in the cache, we need to split the memory address into three parts: the **tag**, the **index**, and the **offset**.

- **Tag**: high-order bits of the address that uniquely identify the block.
- **Index**: bits that are used to index into the cache table.
- **Offset**: bits that determine the position of the data _within_ the block.

<blockquote class="important">

The **cache offset** is different from the **page offset**. The **cache offset** is used to access data within a cache block, while the **page offset** is used to access data within a page.

</blockquote>

### **Cache Formulas**

<blockquote class="equation">

$$
\begin{align*}
&m && \text{ memory address bits} \\
&s && \text{ index bits} \\
&b && \text{ offset bits} \\
&t = m - (s + b) && \text{ tag bits} \\
\\
&S=2^s && \text{ number of sets} \\
&B=2^b && \text{ block size} \\
&C=B \times S && \text{ cache size} \\
\end{align*}
$$

</blockquote>

<blockquote class="important">

The **cache size** is the product of the **block size** and the **number of sets**. It does **NOT** include the size of the tag or other metadata.

</blockquote>

<details><summary>Exercise: Cache Formulas</summary>

Suppose 64-byte blocks, 128 set cache. Calculate the number of bits for the tag, index, and offset.

We first calculate the number offset bits based on the size of the cache block:

$$
\text{offset bits} = b = \log_2(64) = 6 \text{ bits}.
$$

Next we calculate the number of index bits based on the set size:

$$
\text{index bits} = s = \log_2(128) = 7 \text{ bits}.
$$

Finally, we calculate the number of tag bits:

$$
\text{tag bits} = t = 32 - (6 + 7) = 19 \text{ bits}.
$$

Which bytes are stored in the same **block** as byte from `0x1037`?

We first convert `0x1037` to binary:

$$
\text{0x1037} = 0001 \; 0000 \; 0011 \; 0111_2.
$$

We can extract the tag, index, and offset bits:

$$
\underbrace{0 \dots 0}_{\text{tag bits}} \; \underbrace{1\;0000\;00}_{\text{index bits}} \; \underbrace{ 11 \;0111}_{\text{offset bits}}.
$$
$$
\begin{align*}
\text{index} &= 1\;0000\;00_2 &&= 128_{10} \\
\text{offset} &= 11\;0111_2 &&= 55_{10}.
\end{align*}
$$

Therefore, the byte from memory address `0x1037` is stored at set `128` and cache block offset `55`. This means that memory addresses that have the same **index bits** have their bytes stored in the same cache block.
</details>



### **Access Pattern**

<blockquote class="definition">

A **cache hit** occurs when the data is found in the cache. A **cache miss** occurs when the data is not found in the cache.

</blockquote>

When accessing the cache, we go through the following steps:

1. Extract the **tag**, **index**, and **offset** bits from the memory address.
2. Use the **index** bits to find the corresponding set in the cache.
3. Check if **valid bit** is set; otherwise it is a **cache miss**.
4. Check if the **tag** in the cache matches the **tag** from the memory address.
4. If the tags match, it is a **cache hit**; otherwise, it is a **cache miss**.

If we have a **cache miss**:
1. Load the **entire block** from the main memory into the cache.
2. Update the **tag** and **valid bit** in the cache.
3. Retrieve the data from the cache.

<blockquote class="important">

To load the entire block from the main memory, we can simply zero out the **offset** bits to find the start memory address, and load the block of data into the cache based on the **block size**.

</blockquote>

If we have a **cache hit**:

1. Use the **offset** bits to find the position of the data within the block.
2. Retrieve the data from the cache.


### **Associative Cache**

In order to avoid having too many cache misses, we can use an **associative cache**. In an associative cache, each index in the cache can map to multiple cache blocks in the same set.

<blockquote class="definition">

An **associative cache** is a type of cache where each index in the cache can map to **multiple** cache blocks in the same set, where a **way** is a "column" in the cache.

</blockquote>

<img src="https://branyang02.github.io/images/associative_cache.png" alt="Associative Cache" style="display: block; max-height: 40%; max-width: 40%;">

<span class="caption">
  In this example 2-way set associative cache, each set contains 2 cache blocks.
</span>

When accessing an associative cache, we follow the same steps as a direct-mapped cache, but we need to check for **multiple** tags in the set to check for a cache hit. If we have a **cache miss** meaning that none of the tags from any of the blocks in the set match the tag from the memory address, we need to evict a block from the set to make room for the new block. If there is an empty block in the set, we can simply load the block from the main memory into the cache.

To calculate the cache size of an associative cache, we need to multiply the **number of sets** by the **number of ways** by the **block size**.

<blockquote class="equation">

$$
\begin{align*}
&E && \text{ number of blocks per set (``ways")} \\
&C = B \times S \times E && \text{ cache size} \\
\end{align*}
$$

</blockquote>



#### **Replacement Policies**

When we have a **cache miss** in an associative cache, we need to decide which block to evict from the set to make room for the new block. This decision is based on the **replacement policy**.

<blockquote class="definition">

**Least Recently Used** (**LRU**) is a replacement policy that evicts the block that has not been used for the longest time.

</blockquote>

One way to implement LRU is use include a `LRU` bit for each set in the associative cache. The `LRU` bit indicates which **way** is the least recently used. When a block is accessed (either read (hits) or write (misses)), the `LRU` bit is updated. When we evict a block from the set, we evict the block that has the `LRU` bit set.

There are aos other replacement policies such as **First-In-First-Out** (**FIFO**), **Least Frequently Used** (**LFU**), and **Random**.






### **Write Policies**

If the value is **NOT** in the cache, we can:
- **write-allocate**: write value to cache, then load the rest of the **entire** block from memory.
- **write-no-allocate**: write directly to memory, DO NOT add to cache.

If the value is in the cache, we can:
- **write-through**: write to cache, and also write to next layers of memory.
- **write-back**: write to cache, and **only** write to next layers of memory when the block is evicted from the cache.

<blockquote class="important">

**Write-back** requires a **dirty bit** to indicate if the block has been modified. If the block is evicted from the cache and the **dirty bit** is set, the block is written back to the next layer of memory.

</blockquote>

<details><summary>Exercise: write-allocate + write-back</summary>


<img src="https://branyang02.github.io/images/cache_exercise.png" alt="Write Allocate" style="display: block; max-height: 50%; max-width: 50%;">

1. First, we need to compute the tag, index, and offset bits for the target memory address `0x04`:

$$
\begin{align*}
\text{offset bits} = b = 1 \text{ bit}, \quad \text{index bits} = s = 1 \text{ bit}, \quad \text{tag bits} = t = 6 \text{ bits}.
\end{align*}
$$

2. Next, we need to find the index bits and offset bits for the memory address `0x04`:

$$
\begin{align*}
\text{0x04} = 0000 \; 0100_2 = \underbrace{0000\;01}_{\text{tag bits}} \; \underbrace{0}_{\text{index bits}} \; \underbrace{0}_{\text{offset bits}}.
\end{align*}
$$

3. We index into the table at set 0, and check if we have a tag match for any of the ways. In this case, we have a **cache miss** since both ways are occupied and the tags do not match.

4. We choose to evict the block at way 1, since the `LRU` bit is set to `1`. However, since the block at way 1 has a **dirty bit** set to 1, we need to perform **write-back** to write the value in the block to next layer of memory before we can evict the block. We can find the corresponding memory address by the current block's tag and index bits:
$$
\begin{align*}
\text{Tag} &= 01\; 1000 \\
\text{Index} &= 0 \\
\end{align*}
$$
Since the size of each cache block contains 2 bytes, we need to write the value back to the memory address at:

$$
\begin{align*}
\text{write-back addr.}_1 &= \text{Tag, Index, Offset} \\
&= 01 \; 1000 \; 0 \; 0 = \text{0x}60 \\
\text{write-back addr.}_2 &= \text{Tag, Index, Offset} \\
&= 01 \; 1000 \; 0 \; 1 = \text{0x}61 \\
\end{align*}
$$

Therefore, we need to write teh values in the current block to memory address `0x60` and `0x61`.

5. After we perform write-back, we need to perform the **write-allocate** policy, where we not only update the value at address `0x04`, but also load the rest of the block into the cache. To find the other memory address that we need to read from, we need to read the value from the same cache block where the offset bit is `1`, thereby reading in the value at address `0x05`.

6. Finally, we **update the cache block** at way 1 with the new tag `000001`, and the new block value `0xFF`, as well as `M[0x05]`. We also set the **dirty bit** to `0`, since the block has not been modified. We also update the `LRU` bit to `0`, since way 0 has been the least recently used.

The final cache table is shown below:

<img src="https://branyang02.github.io/images/cache_exercise_solution.png" alt="Write Allocate Solution" style="display: block; max-height: 50%; max-width: 50%;">

</details>

<blockquote class="important">

When performing **write-allocate + write-back**, if we write to an _unallocated_ block, we need to read the entire block from memory, update the block, and set the **dirty bit** to `1`.

</blockquote>



#### **Faster Writes**

A **write buffer** is a type of data buffer that can be used to hold data being written from the cache to main memory or to the next cache in the memory hierarchy to improve performance and reduce latency.


### **TLB**

The **Translation Lookaside Buffer** (**TLB**) is a cache that stores the most recent translations from virtual memory to physical memory. The TLB is used to speed up the translation process by avoiding the need to access the page table in memory.

<blockquote class="definition">

a **TLB** is a cache that stores the most recent translations from virtual memory to physical memory.

</blockquote>

A TLB functions similarly to a regular cache, but it stores different data

<div class="small-table">

|Typical L1 Cache|TLB|
|----------------|---|
|physical address | VPN|
|bytes from memory|**last-level** PTE|
|tens of bytes per block | **one** PTE per block|

</div>

Therefore, given a **VPN**, we can find the corresponding **PTE** in the TLB without going through the translation process in the page table. We can think of it as a function that takes a **VPN** as input and returns the corresponding **PTE**.

$$
\text{TLB}: \text{VPN} \rightarrow \text{PTE}
$$

<blockquote class="important">

Since we only store **one** PTE per block, we always have $0$ bits for the **cache offset**. Therefore, the **VPN** is split into the **tag** and **index** bits.

</blockquote>

On a **TLB hit**, meaning we have a corresponding PTE given a VPN, we can simply extract the PTE can check for permission bits before getting the PPN. On a **TLB miss**, we need to access the page table in memory to find the corresponding PTE, and then update the TLB with the new translation.


<blockquote class="important">

In a multi-level page table setup, we use the **entire** VPN to index into the TLB, not the individual parts of the VPN that correspond to each level of the page table.

</blockquote>




## **Threads**

- Motivation
  - **concurrency**: perform multiple tasks can be executed simultaneously.
  - **parallelism**: perform multiple tasks with _more resources_.

We can run multiple threads in a single process to achieve concurrency and parallelism. Threads in the same process share the same **virtual memory space**, but have their own **registers** and **stack**.

<img src="https://branyang02.github.io/images/single_vs_multi_threads.png" alt="Threads" style="display: block; max-height: 50%; max-width: 50%;">

### **Thread Creation**

We can use the `pthread_create` function to create a new thread in C. Below is a typical pattern for creating a new thread:

```c
void *ComputePi(void *argument) { ... }
void *PrintClassList(void *argument) { ... }
int main() {
  pthread_t pi_thread, list_thread;
  if (0 != pthread_create(&pi_thread, NULL, ComputePi, NULL)) {
    handle_error();
  }
  if (0 != pthread_create(&list_thread, NULL, PrintClassList, NULL)) {
    handle_error();
  }
  // more code in the main function
}
```

In this example, we have two functions `ComputePi` and `PrintClassList` that perform different tasks on different threads.

`pthread_create` takes four arguments:
- `pthread_t *thread`: a pointer to a `pthread_t` variable that stores the thread ID.
- `const pthread_attr_t *attr`: a pointer to a `pthread_attr_t` variable that specifies the thread attributes.
- `void *(*start_routine)(void *)`: a function pointer that points to the function that the thread will execute.
- `void *arg`: an argument that is passed to the function that the thread will execute.

When `pthread_create` is called, the OS creates a new thread and executes the function `start_routine` with the argument `arg`. The thread runs _concurrently_ with the main thread. When the specified function returns, the thread is terminated.

<blockquote class="important">

Returning from the `main` function ends the **entire** process, killing **all** threads in the process.

</blockquote>


<blockquote class="important">

The function that `pthread_create` points to must have the following signature:

```c
void *function_name(void *argument);
```

</blockquote>

<details open><summary>Example: Thread Creation</summary>

We can create a new thread that prints "Hello, World!" using the following code:

```execute-c
#include <pthread.h>
#include <stdio.h>

void *PrintHello(void *arg) {
    printf("Hello, World from thread!\n");
    return NULL;
}

int main() {
    pthread_t thread_id;
    if (0 != pthread_create(&thread_id, NULL, PrintHello, NULL)) {
        perror("pthread_create");
        return 1;
    }
    printf("Hello, World from main!\n");
    return 0;
}
```

In this example, we create a new thread using `pthread_create`, and continue to execute the main thread. However, we see that the output for this program is **non-deterministic**. Sometimes the output from the main thread is printed first, and sometimes the output from the new thread is printed first.

This is because the main thread could finish before the new thread finishes, causing the **entire** process to terminate.


</details>











### **Race Conditions**

From the example above, we see that the output of the program is **non-deterministic**. This is because the two threads are **racing** to print their respective messages. This is known as a **race condition**.

<img src="https://i.ytimg.com/vi/KGnXr62bgHM/maxresdefault.jpg" alt="Race Conditions" style="display: block; max-height: 70%; max-width: 70%;">

<span class="caption">
  Source: <a href="https://dennylesmana.medium.com/what-is-race-condition-d678f87af108">What is race condition?
</a>
</span>

We can use `pthread_join` to wait for a thread to finish before continuing execution in the main thread. `pthread_join` takes two arguments:
- `pthread_t thread`: the thread ID of the thread to wait for.
- `void **value_ptr`: a pointer to a variable that stores the return value of the thread.

<details open><summary>Example: pthread_join</summary>

Now, we want to wait for the new thread to finish before return in the `main` function. We use `pthread_join(thread_id, NULL)` to wait for the new thread to finish before returning.

```execute-c
#include <pthread.h>
#include <stdio.h>

void *PrintHello(void *arg) {
    printf("Hello, World from thread!\n");
    return NULL;
}

int main() {
    pthread_t thread_id;
    if (0 != pthread_create(&thread_id, NULL, PrintHello, NULL)) {
        perror("pthread_create");
        return 1;
    }
    printf("Hello, World from main!\n");
    pthread_join(thread_id, NULL);
    return 0;
}
```

Now, the output of the program prints the result from both threads.

</details>

### **Passing Arguments to Threads**


To pass in multiple arguments to the funciton, we can create a `struct` that contains all the arguments, and pass in a pointer to the `struct` as the argument. Below is an example:

```c
int values[1024];
struct ThreadInfo {
  int start, end, result;  // start and end index of the array, and the result
};
void *sum_thread(void *arg) {
  struct ThreadInfo *info = (struct ThreadInfo *) arg;
  int sum = 0;
  for (int i = info->start; i < info->end; i++) {
    sum += values[i];  // sum the values in the array from `start` to `end`
  }
  info->result = sum;  // store the result in the `result` field
  return NULL;
}
int sum_all() {
  pthread_t thread[2];
  struct ThreadInfo info[2];
  for (int i = 0; i < 2; i++) {
    info[i].start = i*512;
    info[i].end = (i+1)*512;
    pthread_create(&thread[i], NULL, sum_thread, &info[i]);
  }
  // wait for threads to finish
  for (int i = 0; i < 2; i++) {
    pthread_join(thread[i], NULL);
  }
  // combine results
  return info[0].result + info[1].result;
}
```

In this example, we create a struct `ThreadInfo` that contains the start and end index of the array that we want to sum for each thread, as well as the `result` of the local sum. In the `sum_all` function, we assign index `0-511` to thread 1 and index `512-1023` to thread 2 to compute the overall sum of the array `values`. We then wait for both threads to finish using `pthread_join`, and combine the results to get the final sum.


<details ><summary>Running the above program</summary>

```execute-c
#include <pthread.h>
#include <stdio.h>

int values[1024];
struct ThreadInfo {
  int start, end, result;  // start and end index of the array, and the result
};

void *sum_thread(void *arg) {
  struct ThreadInfo *info = (struct ThreadInfo *) arg;
  int sum = 0;
  for (int i = info->start; i < info->end; i++) {
    sum += values[i];  // sum the values in the array from `start` to `end`
  }
  info->result = sum;  // store the result in the `result` field
  return NULL;
}

int sum_all() {
  pthread_t thread[2];
  struct ThreadInfo info[2];
  for (int i = 0; i < 2; i++) {
    info[i].start = i*512;
    info[i].end = (i+1)*512;
    pthread_create(&thread[i], NULL, sum_thread, &info[i]);
  }
  // wait for threads to finish
  for (int i = 0; i < 2; i++) {
    pthread_join(thread[i], NULL);
  }
  // combine results
  return info[0].result + info[1].result;
}

int main() {
  for (int i = 0; i < 1024; i++) {
    values[i] = i*i;
  }
  printf("Sum of values (2 threads): %d\n", sum_all());

  int sum = 0;
  for (int i = 0; i < 1024; i++) {
    sum += values[i];
  }
  printf("Sum of values (1 thread): %d\n", sum);
  return 0;
}
```

</details>


### **Synchronization**

When multiple threads access shared resources, we need to ensure that the threads do not interfere with each other when reading or writing to the shared resources. One way to achieve this is to use only **atomic operations** to access shared resources.

<blockquote class="definition">

An **atomic operation** is an operation that is _indivisible_, meaning it cannot be interrupted by another thread. An atomic operation is either fully completed or not started at all.

</blockquote>

However, not all operations can be made _atomic_. We introduce the idea of **mutual exclusion** to ensure that only one thread can access a shared resource at a time.

#### **Mutual Exclusion**

<blockquote class="definition">

**Mutual exclusion** is a property that ensures that only one thread can access a shared resource at a time.

</blockquote>

We call code that only **ONE** thread could execute at a time a **critical section**. We can use **locks** to protect the critical section.

### **Locks**

<blockquote class="definition">

A **lock** is a synchronization primitive (object) that provides **mutual exclusion** to shared resources.

</blockquote>

the lock has two operations:

- **lock**: _wait_ until the lock is free, then acquire the lock, preventing other threads from accessing the shared resource.
- **unlock**: release the lock, allowing other threads to access the shared resource.

<blockquote class="important">

A thread must **wait** until the lock is available before it can acquire the lock. The thread will not execute other code while waiting for the lock.

</blockquote>


**`pthread_mutex_t`** is a type of lock that is used to provide mutual exclusion in C. Below is an example of using a `pthread_mutex_t` to protect a critical section:

```c
#include <pthread.h>

pthread_mutex_t lock;  // declare a lock
pthread_mutex_init(&lock, NULL);  // initialize the lock

void *thread_function(void *arg) {
  pthread_mutex_lock(&lock);  // acquire the lock
  // critical section
  pthread_mutex_unlock(&lock);  // release the lock
  return NULL;
}
```

We use `pthread_mutex_lock` to acquire the lock before entering the critical section, and `pthread_mutex_unlock` to release the lock after exiting the critical section. We can use `pthread_mutex_destroy` to destroy the lock after all threads have finished using it.

<details><summary>Example: Using pthread_mutex_t</summary>


```execute-c
#include <pthread.h>
#include <stdio.h>

int shared_resource = 0;
pthread_mutex_t lock;

void *increment(void *arg) {
    pthread_mutex_lock(&lock);
    for (int i = 0; i < 10000; i++) {
        shared_resource++;
    }
    pthread_mutex_unlock(&lock);
    return NULL;
}

int main() {
    int n = 50;
    pthread_t thread[n];
    pthread_mutex_init(&lock, NULL);
    for (int i = 0; i < n; i++) {
        pthread_create(&thread[i], NULL, increment, NULL);
    }
    for (int i = 0; i < n; i++) {
        pthread_join(thread[i], NULL);
    }
    printf("Shared resource: %d\n", shared_resource);
    pthread_mutex_destroy(&lock);
    return 0;
}
```


In this example, we have a shared resource `shared_resource` that is incremented by $n$ threads. We use a `pthread_mutex_t` lock to protect the critical section where the shared resource is incremented. We initialize the lock using `pthread_mutex_init`, and destroy the lock using `pthread_mutex_destroy` after all threads have finished.

</details>

**`pthread_barrier_t`**: a synchronization primitive that allows multiple threads to wait for each other at a certain point in the program.

A barrier has a wait operation:
- `pthread_barrier_wait`: a thread waits at the barrier until **ALL** threads have reached the barrier.

Instead of using acquiring and releasing locks for every shared resources that are computed in the critical section, we can use a barrier to ensure that all threads have reached a certain point in the program before continuing.

To use a `pthread_barrier_t`, we need to initialize the barrier using `pthread_barrier_init`, and wait for all threads to reach the barrier using `pthread_barrier_wait`. We can destroy the barrier using `pthread_barrier_destroy` after all threads have finished using it.

The pattern is as follows:

```c
pthread_barrier_t barrier;
pthread_barrier_init(&barrier, NULL, n);  // initialize the barrier with `n` threads

void *thread_function(void *arg) {
  // do some work
  pthread_barrier_wait(&barrier);  // wait for all threads to reach the barrier
  // continue with the program
  return NULL;
}
```


<details><summary>Example: Using pthread_barrier_t</summary>

```execute-c
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>

pthread_barrier_t barrier;

struct thread_args {
    int id;
};

void *increment(void *arg) {
    struct thread_args *args = (struct thread_args *) arg;
    printf("Thread %d: before barrier\n", args->id);
    pthread_barrier_wait(&barrier);
    return NULL;
}

int main() {
    int n = 5;
    pthread_t thread[n];
    pthread_barrier_init(&barrier, NULL, n);
    struct thread_args args[n];
    for (int i = 0; i < n; i++) {
        args[i].id = i;
        pthread_create(&thread[i], NULL, increment, &args[i]);
    }
    for (int i = 0; i < n; i++) {
        pthread_join(thread[i], NULL);
    }
    pthread_barrier_destroy(&barrier);
    return 0;
}
```

In this example, we have `n` threads that wait at the barrier before continuing with the program. We use `pthread_barrier_init` to initialize the barrier with `n` threads, and `pthread_barrier_wait` to wait for all threads to reach the barrier. After all threads have reached the barrier, the program continues.

</details>


#### **Deadlocks**

Although locks are useful for protecting shared resources, they can also lead to **deadlocks** if not used correctly.

<blockquote class="definition">

A **deadlock** occurs when two or more threads are waiting for each other to release a lock, causing all threads to be blocked indefinitely.

</blockquote>

Suppose we have the following scenario:

```c
pthread_mutex_t lock1, lock2;
pthread_mutex_init(&lock1, NULL);
pthread_mutex_init(&lock2, NULL);

void *thread1(void *arg) {
  pthread_mutex_lock(&lock1);
  pthread_mutex_lock(&lock2);
  // critical section
  pthread_mutex_unlock(&lock2);
  pthread_mutex_unlock(&lock1);
  return NULL;
}

void *thread2(void *arg) {
  pthread_mutex_lock(&lock2);
  pthread_mutex_lock(&lock1);
  // critical section
  pthread_mutex_unlock(&lock1);
  pthread_mutex_unlock(&lock2);
  return NULL;
}
```

In this example, `thread1` acquires `lock1` first, and then tries to acquire `lock2`. At the same time, `thread2` acquires `lock2` first, and then tries to acquire `lock1`. If `thread1` and `thread2` run concurrently, they can cause a **deadlock** if `thread1` acquires `lock1` and `thread2` acquires `lock2` at the same time.

<div class="small-table">

|Thread 1|Thread 2|
|--------|--------|
|acquire `lock1`   | acquire `lock2`   |
| attempt to acquire `lock2` | attempt to acquire `lock1` |

</div>

<span class="caption">
  We reach a deadlock when thread1 acquires lock1 and thread2 acquires lock2 at the same time.
</span>

A **deadlock** has 4 necessary conditions:
1. **Mutual Exclusion**: At least one resource must be held in a non-sharable mode.
2. **Hold and Wait**: A process must be holding at least one resource and waiting to acquire additional resources held by other processes.
3. **No Preemption**: Resources cannot be preempted from a process holding them.
4. **Circular Wait**: A cycle must exist in the resource allocation graph.


#### **Avoiding Deadlocks vs. Breaking Deadlocks**

When we are in a deadlock, we can break the deadlock by following these strategies:
- **Lock Timeout**: Use a timeout when acquiring locks to prevent indefinite waiting.

To avoid deadlocks, we can follow these strategies:
- **Lock Ordering**: Always acquire locks in the same order to prevent circular waits.
- **Lock Hierarchy**: Assign a unique ID to each lock and acquire locks in increasing order of ID.


### **Monitors**

A **monitor** is a high-level synchronization construct that provides a way to ensure that only one thread can execute a critical section at a time.

A monitor consists of:
- **lock**: a lock that protects the monitor.
- **shared data**: data that is shared between threads.
- **condition variables**: a list of threads that are waiting for a certain condition to be true.

<div class="xsmall-table">

| A Monitor|
|----------|
| Lock     |
| Shared Data |
|condvar 1|
|condvar 2|
|...|
| `operation1()` |
| `operation2()` |

</div>

We can have 0 or more condition variables in a monitor.

<blockquote class="important">

A monitor **must acquire the lock** before accessing the shared data and performing conditional variable operations.

</blockquote>

**conditional variable** operations:
- `wait(cv, lock)`
  - 1. Unlock lock
  - 2. Add **current** thread to the cv queue.
  - 3. **Reacquire** lock before returning
- `broadcast(cv)`
  - remove **all** threads from the cv queue.
- `signal(cv)`
  - remove **one** thread from the cv queue.


<blockquote class="important">

When a thread is in a condition variable queue, it is **not** holding the lock. It is in a **blocked** state, waiting to be **signaled** or **broadcasted**.

</blockquote>

Suppose we have the following setup:

```c
pthread_mutex_t lock;
bool finished = false;
pthread_cont_t finished_cv;

void WaitForFinished() {
  pthread_mutext_lock(&lock);  // always acquire the lock first
  while (!finished) {
    pthread_cond_wait(&finished_cv, &lock);  // wait for the condition to be true
  }
  pthread_mutex_unlock(&lock);  // always release the lock before returning
}

void Finish() {
  pthread_mutex_lock(&lock);  // always acquire the lock first
  finished = true;
  pthread_cond_signal(&finished_cv);  // signal the condition variable
  pthread_mutex_unlock(&lock);
}
```

We have a single shared resource `finished` that is protected by the `lock`. Suppose we have 2 threads `thread1` and `thread2` that call `WaitForFinished` and `Finish` respectively.

If `Finish` is called before `WaitForFinished`, `Finish` will set `finished` to `true` and unlock the `lock`. When `WaitForFinished` is called, it will acquire the `lock`, and check that `finished` is set to true, therefore it skips the `while` loop and continues with the program to release the `lock`.

However, if `WaitForFinished` is called first, it will acquire the `lock`, and check that `finished` is set to false, therefore it will enter the `while` loop and `pthread_cond_wait` for the condition to be true. When `pthread_cond_wait` is called, `thread1` that calls the `WaitForFinished` will:

1. **unlock** the `lock` that it acquired in the beginning of the funciton.
2. Add itself (`thread1`) to the `finished_cv` queue.

In the mean time, `thread2` that calls the `Finish` has been waiting for the `lock` to be released. Once it acquires the `lock`, it sets `finished` to `true`, and calls `pthread_cond_signal` to **release all threads** in the `finished_cv` queue, including `thread1`.

Now, back in `thread1`, it will return from the `pthread_cond_wait` and **reacquire** the `lock`, once `thread2` releases the `lock`. Since we are still in the `while` loop, `thread1` will check that `finished` is set to `true`, and continue with the program. Finally, `thread1` will release the `lock` before returning.


#### **Producer Consumer Pattern**

The **producer/consumer** pattern is a synchronization pattern where one or more threads produce data and put it into a **shared buffer**, while one or more threads consume the data from the shared buffer.


```c
pthread_mutex_t lock;  // lock to protect the buffer
pthread_cond_t data_ready; pthread_cond_t space_ready;  // condition variables
BoundedQueue buffer;  // shared buffer

Produce(item) {
  pthread_mutex_lock(&lock);  // acquire the lock
  while (buffer is full) {
    pthread_cond_wait(&space_ready, &lock);  // wait for space in the buffer
  }
  // Once there is space in the buffer, add the item
  buffer.add(item);
  pthread_cond_signal(&data_ready);  // signal that there is data in the buffer
  pthread_mutex_unlock(&lock);
}

Consume() {
  pthread_mutex_lock(&lock);  // acquire the lock
  while (buffer is empty) {
    pthread_cond_wait(&data_ready, &lock);  // wait for data in the buffer
  }
  // Once there is data in the buffer, remove the last item from the queue
  item = buffer.dequeue();
  pthread_cond_signal(&space_ready); // signal that there is space in the buffer
  pthread_mutex_unlock(&lock);
  return item;
}
```


<details><summary>Example: Producer Consumer Pattern</summary>

```execute-c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define BUFFER_SIZE 10

pthread_mutex_t lock;
pthread_cond_t data_ready, space_ready;
int buffer[BUFFER_SIZE];
int in = 0, out = 0;

void Produce(int item) {
    pthread_mutex_lock(&lock);
    while ((in + 1) % BUFFER_SIZE == out) {
        pthread_cond_wait(&space_ready, &lock);
    }
    buffer[in] = item;
    in = (in + 1) % BUFFER_SIZE;
    pthread_cond_signal(&data_ready);
    pthread_mutex_unlock(&lock);
}

int Consume() {
    pthread_mutex_lock(&lock);
    while (in == out) {
        pthread_cond_wait(&data_ready, &lock);
    }
    int item = buffer[out];
    out = (out + 1) % BUFFER_SIZE;
    pthread_cond_signal(&space_ready);
    pthread_mutex_unlock(&lock);
    return item;
}

void *Producer(void *arg) {
    for (int i = 0; i < 100; i++) {
        Produce(i);
        printf("Produced: %d\n", i);
    }
    return NULL;
}

void *Consumer(void *arg) {
    for (int i = 0; i < 100; i++) {
        int item = Consume();
        printf("             Consumed: %d\n", item);
    }
    return NULL;
}

int main() {
    pthread_t producer, consumer;
    pthread_mutex_init(&lock, NULL);
    pthread_cond_init(&data_ready, NULL);
    pthread_cond_init(&space_ready, NULL);
    pthread_create(&producer, NULL, Producer, NULL);
    pthread_create(&consumer, NULL, Consumer, NULL);
    pthread_join(producer, NULL);
    pthread_join(consumer, NULL);
    pthread_mutex_destroy(&lock);
    pthread_cond_destroy(&data_ready);
    pthread_cond_destroy(&space_ready);
    return 0;
}
```

In this example, we have a producer that produces items from `0` to `99`, and a consumer that consumes the items from the shared buffer. We use a `pthread_mutex_t` lock to protect the shared buffer, and two condition variables `data_ready` and `space_ready` to signal when there is data in the buffer and when there is space in the buffer respectively.

</details>

















## **Networking**

coming soon...

## **Secure Channels**

Secure communication over an insecure channel involves safeguarding the data against unauthorized access and manipulation. The two primary concerns are confidentiality and authenticity, ensuring the data remains private and verifiable respectively.

### **Example Attack Scenarios**

- **Passive Attacks:**
  - **Eavesdropping:** An unauthorized party listens to the communications between Machine A and Machine B.
- **Active Attacks:**
  - **Machine-in-the-middle Attack:** An attacker intercepts, alters, or injects messages between Machine A and Machine B.

<img src="https://images.shiksha.com/mediadata/ugcDocuments/images/wordpressImages/2023_02_MicrosoftTeams-image-304.jpg" alt="Secure Channels" style="display: block; max-height: 70%; max-width: 70%;">
<span
    class="caption"> Source: <a href="https://www.shiksha.com/online-courses/articles/difference-between-active-and-passive-attacks/">Difference Between Active and Passive Attacks</a>

</span>

### **Confidentiality**

- Confidentiality ensures that the message is only readable by the intended recipient.

**Example Scenario:**

- Machine A sends a message to Machine B.
- Machine M, acting as a machine-in-the-middle, intercepts and may alter the message pretending to be Machine B.

#### **Symmetric Encryption Functions**

Symmetric encryption uses a shared secret key for both encrypting and decrypting messages.

- **Encrypt Function:**
  - $E(key, message) =$ `ciphertext`
- **Decrypt Function:**
  - $D(key, ciphertext) =$ `message`

**Using Symmetric Encryption**

If Machine A and Machine B have a shared secret, the communication process involves:

1. A computes the ciphertext using the shared key: $E(key, message)$.
2. A sends the ciphertext to B.
3. B decrypts the received ciphertext: $D(key, ciphertext)$.

Here is an example of symmetric encryption in C:

```execute-c
#include <stdio.h>
#include <string.h>

void encrypt(char *message, char *key) {
    for (int i = 0; i < strlen(message); i++) {
        message[i] += key[i % strlen(key)];
    }
}

void decrypt(char *message, char *key) {
    for (int i = 0; i < strlen(message); i++) {
        message[i] -= key[i % strlen(key)];
    }
}

int main() {
    char message[] = "Hello, World!";
    char key[] = "secret";

    printf("Original message: %s\n", message);

    encrypt(message, key);
    printf("ciphertext: %s\n", message);

    decrypt(message, key);
    printf("Decrypted message: %s\n", message);

    return 0;
}
````

#### **Asymmetric Encryption Functions**

Asymmetric encryption uses a pair of keys: a public key for encryption and a private key for decryption. The public key is shared, while the private key is kept secret.

- **Encrypt Function:**
  - $PE_{\text{public}}(key_{\text{public}}, message) =$ `ciphertext`
- **Decrypt Function:**
  - $PD_{\text{private}}(key_{\text{private}}, ciphertext) =$ `message`

**Using Asymmetric Encryption**

1. B generates a pair of keys: $key_{\text{public}}, key_{\text{private}}$.
2. B sends the public key $key_{\text{public}}$ to A.
3. A computes the ciphertext using the public key: $PE_{\text{public}}(key_{\text{public}}, message)$.
4. A sends the ciphertext to B.
5. B decrypts the received ciphertext using the private key: $PD_{\text{private}}(key_{\text{private}}, ciphertext)$.

<details><summary>Example Asymmetric Encryption in C</summary>

The following code implements asymmetric encryption in C, however, it is a simplified version of asymmetric encryption and it is not secure.

```execute-c
#include <stdio.h>
#include <string.h>

int e(int msg, int key) {
    return msg * key;
}
int d(int msg, int key) {
    return msg * key;
}

int main() {
    int message = 0xdeadbeef;
    int k1 = 2501, k2 = 3221654797;

    printf("-> message: %x; ciphertext: %x; decrypted: %x\n",
        message, e(message, k1), d(e(message, k1), k2));
    printf("<- message: %x; ciphertext: %x; decrypted: %x\n",
        message, e(message, k2), d(e(message, k2), k1));

    return 0;
}

```

In this example, we assumed that `k1` and `k2` are already given. In a real-world scenario, `k1` and `k2` would be generated using a secure algorithm.

</details>

### **Authenticity**

- Authenticity ensures the message is actually sent by the claimed sender.

#### **Message Authentication Code (MAC)**

MAC is used to verify the integrity and the authenticity of a message. It also ensures that the message has not been tampered with during transmission.

Supoose we have a funtion `MAC` that is given by the _expert_:

- **MAC Function:**
  - $MAC(key, message) = tag$
- **MAC Verification Function:**
  - $MAC_{\text{verify}}(key, message, tag)$ = `true` or `false`

**Using MAC**

If Machine A and Machine B have a shared secret, the communication process involves:

1. A computes the MAC tag using the shared key: $MAC(key, message)$.
2. A sends the message and the MAC tag to B.
3. B verifies the MAC tag: $MAC_{\text{verify}}(key, message, tag)$.
4. If the MAC tag is verified, B accepts the message.

**Using MAC with Encryption**

1. A computes the ciphertext using the shared key: $E(key, message)$.
2. A computes the MAC tag using the shared key: $MAC(key, ciphertext)$.
3. A sends the ciphertext and the MAC tag to B.
4. B verifies the MAC tag: $MAC_{\text{verify}}(key, ciphertext, tag)$.
5. If the MAC tag is verified, B decrypts the ciphertext: $D(key, ciphertext)$.

<details><summary>Example MAC + Symmetric Encryption in C</summary>

The following code implements a simple MAC in C:

```execute-c
#include <stdio.h>
#include <string.h>

void encrypt(char *message, char *key) {
    for (int i = 0; i < strlen(message); i++) {
        message[i] += key[i % strlen(key)];
    }
}

void decrypt(char *message, char *key) {
    for (int i = 0; i < strlen(message); i++) {
        message[i] -= key[i % strlen(key)];
    }
}

void mac(char *message, char *key, char *tag) {
    for (int i = 0; i < strlen(message); i++) {
        tag[i] = message[i] ^ key[i % strlen(key)];
    }
}

int mac_verify(char *message, char *key, char *tag) {
    char computed_tag[strlen(message)];
    mac(message, key, computed_tag);
    return memcmp(tag, computed_tag, strlen(message)) == 0;
}

int main() {
    char message[] = "Hello, World!";
    char key[] = "secret";
    char tag[strlen(message)];

    printf("Original message: %s\n", message);

    encrypt(message, key);
    printf("ciphertext: %s\n", message);

    mac(message, key, tag);
    printf("MAC tag: %s\n", tag);

    if (mac_verify(message, key, tag)) {
        printf("MAC tag verified.\n");
        decrypt(message, key);
        printf("Decrypted message: %s\n", message);
    } else {
        printf("MAC tag not verified.\n");
    }

    return 0;
}
```

</details>

#### **Digital Signatures**

Digital signatures are used to verify the authenticity of a message and ensure that the message has not been tampered with.

- **Sign Function:**
  - $S(key_{\text{private}}, message) = signature$
- **Verify Function:**
  - $V(key_{\text{public}}, message, signature)$ = `true` or `false`

**Using Digital Signatures**

1. A generates a pair of keys: $key_{\text{public}}, key_{\text{private}}$.
2. A computes the signature using the private key: $S(key_{\text{private}}, message)$.
3. A sends the message and the signature to B.
4. B verifies the signature using the public key: $V(key_{\text{public}}, message, signature)$.

### **Handling Replay Attacks**

Replay attacks involve an attacker intercepting a message and replaying it at a later time.

- **Replay Attack Scenario:**
  - Machine A sends a message to Machine B.
  - Machine M intercepts the message and sends it to B again.

**Nonces** are used to prevent replay attacks.

A nonce is a number used only once in a cryptographic communication. It is used to prevent replay attacks.

- **Example Scenario:**
  - Machine A sends a message to Machine B.
  - Machine M intercepts the message and sends it to B again.

**Using Nonces**

1. A generates a nonce: $N_A$.
2. A sends the message and the nonce to B.
3. B verifies the nonce and accepts the message. B generates a nonce: $N_B$.

### **Certificate**

Certificates are used to verify the authenticity of a public key.

- **Certificate Authority (CA):**
  - A trusted third party that issues certificates.
- **Certificate:**
  - Contains the public key and information about the owner.
- **Certificate Verification:**
  - Ensures the certificate is valid and issued by a trusted CA.

**Structure of a Certificate**

- **subject:** the entity the certificate is about.
- **issuer:** the entity that issued the certificate.
- **public key:** the public key of the subject.
- **signature:** the signature of the issuer.

**Public Key Infrastructure (PKI)**

- **Certificate Authority (CA):**
  - Issues certificates.
- **Registration Authority (RA):**
  - Verifies the identity of the certificate holder.
- **Certificate Repository:**
  - Stores certificates.

<details><summary>Example Certificate Scenario</summary>

- **Problem**:

  - Machine A wans to send a message to Machine B.
  - Machine A does not have Machine B's public key, however, Machine A trusts the CA (A has the CA's public key).

- **Setup**:

  1. CA can issue a certificate for Machine B:
     - **subject:** Machine B
     - **issuer:** CA
     - **public key:** Machine B's public key
     - **signature:** CA's signature
  2. CA sends the certificate to Machine B.

- **Process**:

  1. Machine A receives Machine B's certificate.
  2. Machine A verifies the certificate using the CA's public key.
  3. Machine A extracts Machine B's public key from the certificate.
  4. Machine A sends the message to Machine B using Machine B's public key.

</details>

## **Pipeline**

A pipeline is a technique used to overlap the execution of multiple instructions. It divides the instruction execution into multiple stages, allowing multiple instructions to be executed simultaneously.

### **Pipeline Stages**

- **Fetch**: fetch the instruction from memory.
- **Decode**: decode the instruction.
- **Execute**: execute the instruction.
- **Memory**: access memory.
- **Writeback**: write the result back to the register file.

In order to hold the data between stages, we use **pipeline registers**.

<img src="https://branyang02.github.io/images/pipeline.png" alt="Pipeline" style="display: block; max-height: 70%; max-width: 70%;">

<span class="caption"> 
The pipeline consists of five stages: Fetch, Decode, Execute, Memory, and Writeback. Each stage has a pipeline register to hold the data between stages.
</span>

We evaluate the performance of a pipeline using the following metrics:

- **Latency**: the time taken to complete a single instruction.
- **Throughput**: the number of instructions completed per unit of time.

<blockquote class="equation">

$$
\begin{equation*}
\text{Latency} = \text{Cycle Time} \times \text{Number of Stages}
\end{equation*}
$$

</blockquote>

<blockquote class="equation">

$$
\begin{equation*}
\text{Throughput} = \frac{\text{Number of instructions}}{\text{Total time to execute all instructions}}
\end{equation*}
$$

</blockquote>

<blockquote class="important">

Note that the total time to execute all instructions is the time taken between the **end** of the first instruction and the **end** of the last instruction. Therefore:

$$
\text{Total time to execute all instructions} = \left(\text{instr}^n_{end} - \text{instr}^1_{end} + 1 \right) \times \text{Cycle Time},
$$

where $\text{instr}^n_{end}$ is the cycle number when the last instruction ends, and $\text{instr}^1_{end}$ is the cycle number when the first instruction ends.

</blockquote>

<details open><summary>Example Pipeline Latency and Throughput</summary>

<img src="https://branyang02.github.io/images/latency.png" alt="Pipeline" style="display: block; max-height: 70%; max-width: 70%;">

In the example above, suppose _cycle time_ is 500 ps, the **latency** is calculated as follows:

$$
\text{Latency} = \text{Cycle Time} \times \text{Number of Stages} = 500 \cdot 5 \text{ ps} = 2500 \text{ ps}
$$

We also compute the **throughput** based on the equation above:

$$
\begin{align*}
\text{Throughput} &= \frac{\text{Number of instructions}}{\left(\text{instr}^2_{end} - \text{instr}^1_{end} + 1 \right) \times \text{Cycle Time}} \\
&= \frac{2 \text{ instructions}}{(5-4 + 1) \times 500 \text{ ps}} \\
&= \frac{1 \text{ instruction}}{500 \text{ ps}}
\end{align*}
$$

To express the throughput in terms of _ps per instruction_, we take the reciprocal of the throughput:

$$
\text{Throughput} = 1 / \frac{1}{500 \text{ ps}} = 500 \frac{\text{ps}}{\text{instruction}}
$$

</details>

We can increase pipeline performance by increasing the number of states. However, we will only see a **diminishing return** as the number of stages increases. This is because that **pipeline registers** often take time to load and store the data.

<img src="https://branyang02.github.io/images/diminishing_return.png" alt="Diminishing Return" style="display: block; max-height: 70%; max-width: 70%;">
<span
    class="caption"> Diminishing Returns: register delays
</span>

Dividing the instruction execution into multiple stages can lead to **pipeline hazards**.

### **Pipeline Hazards**

<blockquote class="definition">

**Data Hazard**: a data dependency between instructions.

</blockquote>

<details open><summary>Data Hazard Example</summary>

Suppose we have the following instructions:

1. `addq %r8, %r9`
2. `addq %r9, %r8`

Suppose we run these instructions in the pipeline configuration:

<img src="https://branyang02.github.io/images/data_hazard.png" alt="Data Hazard" style="display: block; max-height: 70%; max-width: 70%;">

We can see that the second instruction depends on the result of the first instruction. This data dependency causes a **data hazard**.

</details>

#### **Data Hazard Solutions**

<blockquote class="definition">

**Stalling**: the _hardware_ inserts a `nop` (no operation) instruction to wait for the data to be available.

</blockquote>

We can also use a _compiler_ to manually insert `nop` instructions to resolve data hazards, but this is less efficient than using hardware to do so.

<details open><summary>Stalling Example</summary>

Suppose we have the following instructions:

1. `addq %r8, %r9`
2. `addq %r9, %r8`

To resolve the data hazard, we can insert `nop` instructions:

1. `addq %r8, %r9`
2. _hardware inserts_ `nop`
3. _hardware inserts_ `nop`
4. `addq %r9, %r8`

This way, the second instruction will run _three_ cycles after the first instruction, allowing the data to be available.

</details>

<blockquote class="definition">

**Forwarding**: the process of passing the result from an earlier instruction's source stage to the destination stage of a later instruction in the _same cycle_.

</blockquote>

Different instructions will have different latencies in the pipeline. For example, a `load` instruction will have a longer latency due to memory access compared to an `add` instruction. Here is a table of instructions and their corresponding data ready stages:

| Instruction type | Example instruction  | Data Ready Stage |
| ---------------- | -------------------- | ---------------- |
| Arithmetic       | `addq %r8, %r9`      | **Execute**      |
| Load             | `movq 0(%rax), %rbx` | **Memory**       |
| Store            | `movq %rbx, 0(%rax)` | **Execute**      |
| Branch           | `jne label`          | **Execute**      |

Note that the instruction after a branch instruction (`jne label`) will need to wait until the branch is resolved at the `execute` stage. This means that we _cannot_ `fetch` the next instruction until the `execute` stage of the branch instruction.

<details><summary>Forwarding Example</summary>

Suppose we have the following instructions:

<img src="https://branyang02.github.io/images/forwarding.png" alt="Forwarding" style="display: block; max-height: 70%; max-width: 70%;">

We follow each instruction through the pipeline stages:

1. `addq %r8, %r9`: first instruction.
2. `subq %r8, %r10`: nothing is forwarded since no value is needed.
3. `xorq %r8, %r9`: `%r9` is forwarded to `decode` from `memory` of instruction 1.
4. `addq %r9, %r8`: `%r9` is forwarded to `decode` from `execute` of instruction 3.

At every cycle, the pipeline checks if the data is available in the later stages and forwards it to the earlier stages if needed. This way, the data is available when needed, and the instructions can be executed without stalling.

Note that at every cycle, the value of a register is only updated **once** from an earlier stage. In the example above at instruction 4, the value of `%r9` is first modified in instruction 1, then forwarded to instruction 3, and finally forwarded to instruction 4. Therefore, instruction 4 only needs to use the value of `%r9` from instruction 3.

</details>

- **Stall + Forwarding**: a combination of stalling and forwarding to resolve data hazards.

<details><summary>Stall + Forwarding Example</summary>

Suppose we have the following instructions:

1. `movq 0(%rax), %rbx`
2. `subq %rbx, %rcx`

and the following pipeline stages:

$$

\begin{array}{cccccccccc}
& \text{0} & \text{1} & \text{2} & \text{3} & \text{4} & \text{5} & \text{6} & \text{7} & \text{8} \\
\text{movq 0(\%rax), \%rbx} & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} & & & & \\
\text{subq \%rbx, \%rcx} & & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} & & & \\
\end{array}


$$

We are performing a load from memory in instruction 1. In this case, the value of `%rbx` is not available until the `memory` stage of instruction 1. Therefore, we need to stall instruction 2 until the value of `%rbx` is available.

However, we can see that the `memory` stage of instruction 1 is in the cycle _after_ the `execute` stage of instruction 2. Therefore, we cannot possibly forward the value from a future cycle. In this case, we need to stall instruction 2 and wait for the value of `%rbx` to be available.

$$

\begin{array}{cccccccccc}
& \text{0} & \text{1} & \text{2} & \text{3} & \text{4} & \text{5} & \text{6} & \text{7} & \text{8} & \text{9} \\
\text{movq 0(\%rax), \%rbx} & \text{F} & \text{D} & \text{E} & \color{red} \text{M} & \text{W} & & & & & \\
\text{subq \%rbx, \%rcx} & & \text{F} & \text{D} & \color{red} \text{D} & \text{E} & \text{M} & \text{W} & & & \\
\end{array}


$$

In this case, we will stall instruction 2's `decode` stage until the value of `%rbx` is available. Once the value is available, we can proceed with the execution of instruction 2 at cycle 3.

</details>

### **Control Hazard**

<blockquote class="definition">

**Control Hazard**: a control dependency between instructions.

</blockquote>

Control hazards occur when the next instruction to execute depends on the result of a previous instruction.

We can resolve control hazards by **stalling** or **branch prediction**.

- **Stalling**: the process of waiting for the target of a branch instruction to be known before proceeding with the next instruction.

<details open><summary>Control Hazard Stalling Example</summary>

Suppose we have the following instructions:

1. `cmpq %r8, %r9`
2. `jne label`
3. `xorq %r10, %r11`
4. `movq %r11, 0(%r12)`

We can perform our standard **FDEMW** pipeline stages:

$$

\begin{array}{ccccccccc}
& \text{0} & \text{1} & \text{2} & \text{3} & \text{4} & \text{5} & \text{6} & \text{7} & \text{8} \\
\text{cmpq \%r8, \%r9} & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} & & & & \\
\text{jne label} & & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} & & & \\
\text{xorq \%r10, \%r11} & & & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} & & \\
\text{movq \%r11, 0(\%r12)} & & & & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} & \\
\end{array}


$$

In the first instruction `cmpq %r8, %r9`, the flag is set at `execute` stage. The second instruction `jne label` depends on the flag set by the first instruction. Therefore, the value of the flag is _forwarded_ to the `decode` stage of the second instruction.

However, the `jne label` instruction is a **control hazard** since we do not know the target of the jump until the `execute` stage of the jump instruction. Therefore, we need to stall the pipeline until the target of the jump is known:

$$

\begin{array}{cccccccccc}
& \text{0} & \text{1} & \text{2} & \text{3} & \text{4} & \text{5} & \text{6} & \text{7} & \text{8} & \text{9} \\
\text{cmpq \%r8, \%r9} & \text{F} & \text{D} & \color{red} \text{E} & \text{M} & \text{W} & & & & \\
\text{jne label} & & \text{F} & \color{red} \text{D} & \text{E} & \text{M} & \text{W} & & & \\
\color{#808080}\text{nop} & & & \color{#808080} \text{F} & \color{#808080} \color{#808080}\text{D} & \color{#808080}\text{E} & \color{#808080}\text{M} & \color{#808080}\text{W} & & \\
\color{#808080}\text{nop} & & & & \color{#808080}\text{F} &\color{#808080}\text{D} & \color{#808080}\text{E} &\color{#808080} \text{M} & \color{#808080}\text{W} & \\
\text{xorq \%r10, \%r11} & & & & & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} \\
\text{movq \%r11, 0(\%r12)} & & & & & & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} \\
\end{array}


$$

We must always make sure that the `execute` stage of the jump instruction is completed before we can proceed with the next `fetch` instruction.

</details>

<blockquote class="definition">

**Branch Prediction**: the process of predicting the target of a branch instruction before it is known.

</blockquote>

<details open><summary>Branch Prediction Example</summary>

Suppose we have the following instructions:

```assembly
        cmpq %r8, %r9
        jne label
        xorq %r10, %r11
        movq %r11, 0(%r12)

label:  addq %r8, %r9
        imul %r13, %r14
```

If we follow the **stalling** solution, we essentially _waste_ 2 cycles waiting for the target of the jump instruction to be known. This can be inefficient, especially if the target of the jump instruction is known most of the time.

Instead, we can use **branch prediction** to predict the target of the jump instruction. If the prediction is correct, we can proceed with the execution of the next instruction without stalling.

In the example above, we **_speculate_** that the jump will **NOT** be taken, and we proceed with the execution of the next two instructions in place of the `nop` instructions.

If our speculation is correct, our pipeline will look like this:

$$

\begin{array}{cccccccccc}
& \text{0} & \text{1} & \text{2} & \text{3} & \text{4} & \text{5} & \text{6} & \text{7} & \text{8} \\
\text{cmpq \%r8, \%r9} & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} & & & & \\
\text{jne label} & & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} & & & \\
\text{xorq \%r10, \%r11} & & & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} & & \\
\text{movq \%r11, 0(\%r12)} & & & & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} & \\
\vdots & & & & & & & & & \\
\end{array}


$$

However, if we speculate _incorrectly_, we can **_squash_** the instructions that were executed and proceed with the correct target of the jump instruction:

$$

\begin{array}{cccccccccc}
& \text{0} & \text{1} & \text{2} & \text{3} & \text{4} & \text{5} & \text{6} & \text{7} & \text{8} & \text{9} \\
\text{cmpq \%r8, \%r9} & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} & & & & \\
\text{jne label} & & \text{F} & \text{D} & \color{green} \text{E} & \text{M} & \text{W} & & & \\
\text{xorq \%r10, \%r11} & & & \text{F} & \color{red} \text{D} & \\
\color{#808080}\text{nop} & & & & & \color{#808080} \text{E} & \color{#808080} \text{M} & \color{#808080} \text{W} & & \\
\text{movq \%r11, 0(\%r12)} & & & & \color {red} \text{F} \\
\color{#808080}\text{nop} & & & & & \color{#808080} \text{D} & \color{#808080} \text{E} & \color{#808080} \text{M} & \color{#808080} \text{W} & \\
\text{addq \%r8, \%r9} & & & & & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} \\
\text{imul \%r13, \%r14} & & & & & & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} \\
\end{array}


$$

In the example above, we know we have speculated incorrectly when we reach the `execute` stage (labeled in green) of the jump instruction. We then **_squash_** the instructions that were executed (labeled in red), and replace them with `nop` instructions.

</details>

We want to correctly predict the target of the jump instruction as much as possible to avoid stalling the pipeline. We can use different **branch prediction algorithms** to predict the target of the jump instruction.

- **Static Prediction**: always predict the same target.
  - **forward not taken, backward taken**: jumps that go to a future instruction are not taken, and jumps that go to a previous instruction are taken.
- **Dynamic Prediction**: predict the target based on the history of the branch.
  - **Branch History Table (BHT)**: a table that stores the history of branches.

<img src="https://branyang02.github.io/images/branch_prediction.png" alt="Branch History Table" style="display: block; max-height: 70%; max-width: 70%;">
<span
    class="caption"> Branch History Table (BHT) stores the history of branches. The table is indexed by the program counter (PC) and stores the history of the branch (taken or not taken).
</span>

Jump instructions are often used in loops, and they may take some time to resolve. To speed up this process, we can use a **branch target buffer (BTB)** to store the target of the jump instruction.

- **Branch Target Buffer (BTB)**: a cache that stores the target of the jump instruction.

### **Beyond Pipelining**

- **Multiple Issue**: executing multiple instructions in parallel.

$$

\begin{array}{cccccccccc}
& \text{0} & \text{1} & \text{2} & \text{3} & \text{4} & \text{5} & \text{6} & \text{7} & \text{8} \\
\text{addq \%r8, \%r9} & \text{F} & \text{D} & \color{red}\text{E} & \text{M} & \text{W} & & & & \\
\text{subq \%r10, \%r11} & \text{F} & \text{D} & \color{red}\text{E} & \text{M} & \text{W} & & & \\
\text{xorq \%r9, \%r11} & & \text{F} &\color{red} \text{D} & \text{E} & \text{M} & \text{W} & & \\
\text{subq \%r10, \%rbx} & & \text{F} &\color{red} \text{D} & \text{E} & \text{M} & \text{W} & \\
\end{array}


$$

In the example above, we can see that the first two instructions can be executed in parallel since they do not have any data dependencies. This is an example of **multiple issue**. However, **hazard handling** becomes more complex when we have multiple instructions executing in parallel. In this example, we need to forward the result from `execute` of the first AND second instruction to the `decode` of the third instruction. We also need to forward the result from `execute` of the third instruction to the `decode` of the fourth instruction.

- **Out-of-Order Execution**: executing instructions out of order to increase performance.

We introduce OOO in the next section.

## **Out-of-Order (OOO)**

To increase the performance of the pipeline, we can execute instructions _out of order_. This allows us to execute independent instructions simultaneously, providing an _illusion_ that work is still done in order, even if they are not in the correct order.

<blockquote class="definition">

**Out-of-order**: A technique used to execute instructions in a different order than they appear in the program.

</blockquote>

### **OOO hazards**

<blockquote class="definition">

A **Read-After-Write (RAW)** data hazard occurs when the pipeline creates the potential for an instruction to read an operand before a prior instruction writes to it.

</blockquote>

Suppose we have the following instructions:

1. `addq %r10, %r8`
2. `movq %r8, (%rax)`
3. `movq $100, %r8`
4. `addq %r13, %r8`

The pipeline stages are as follows:

$$

\begin{array}{cccccccccc}
& \text{0} & \text{1} & \text{2} & \text{3} & \text{4} & \text{5} & \text{6} & \text{7} & \text{8} & \text{9} \\
\text{addq \%r10, \%r8} & \text{F} & & & & & \text{D} & \color{red}\text{E} & \text{M} & \text{W} \\
\color{#808080}\text{movq \%r8, (\%rax)} & & \text{F} & & & & & \color{#808080}\text{D} & \color{#808080}\text{E} & \color{#808080}\text{M} & \color{#808080}\text{W} \\
\text{movq \$100, \%r8} & & & \text{F} & \text{D} & \color{green}\text{E} & \text{M} & \text{W} & & \\
\text{addq \%r13, \%r8} & & & & \text{F} & & & \color{red}\text{D} & \text{E} & \text{M} & \text{W} & \\
\end{array}


$$

In the example above `movq $100, %r8` is executed out-of-order. However, when we execute the next instruction `addq %r13, %r8`, we have a **RAW hazard** since its `decode` stage will attempt to fetch from the forward value of `%r8` from the `execute` stage of the `addq %r10, %r8` instruction.

### **Register Version Tracking**

A simple solution to the RAW hazard is to add _version numbers_ to the registers. This way, we can track which version of the register is being used by the instruction. In the example above, we can perform the following steps to resolve the RAW hazard:

$$

\begin{array}{ccccccccc}
& \text{0} & \text{1} & \text{2} & \text{3} & \text{4} & \text{5} & \text{6} & \text{7} & \text{8} & \text{9} \\
\text{addq \%r10}, \color{red}\text{\%r8}_{v1} \to \text{\%r8}_{v2} & \text{F} & & & & & \text{D} & \text{E} & \text{M} & \text{W} \\
\color{#808080}\text{movq \%r8, (\%rax)} & & \text{F} & & & & & \color{#808080}\text{D} & \color{#808080}\text{E} & \color{#808080}\text{M} & \color{#808080}\text{W} \\
\text{movq \$100}, \color{red}\text{\%r8}_{v2} \to \text{\%r8}_{v3} & & & \text{F} & \text{D} & \text{E} & \text{M} & \text{W} & & \\
\text{addq \%r13}, \color{red}\text{\%r8}_{v3} \to \text{\%r8}_{v4} & & & & \text{F} & & & \text{D} & \text{E} & \text{M} & \text{W} & \\
\end{array}


$$

We can also run into issues where we need to keep copies of multiple versions of the register. For example, if we have the following instructions:

1. `addq %r10, %r8`
2. `movq %r8, (%rax)`
3. `movq %r11, %r8`
4. `movq %r8, 8(%rax)`
5. `movq $100, %r8`
6. `addq %r13, %r8`

We want to ensure that instruction `2` uses the value of `%r8` from instruction `1`, and instruction `4` uses the value of `%r8` from instruction `3`, and instruction `6` uses the value of `%r8` from instruction `5`. This is a **write-after-write (WAW) hazard**. We can resolve this by having multiple versions of the register `%r8`.

<blockquote class="definition">

A **Write-After-Write (WAW)** data hazard occurs when the pipeline creates the potential for an instruction to write to a register before a prior instruction writes to it.

</blockquote>

We can resolve WAW hazards by using **register renaming**.

### **Register Renaming**

Register renaming is a technique that involves renaming the registers in the instruction to avoid data hazards.

<blockquote class="definition">

**Register Renaming**: rename _**architectural**_ registers (reg. in assembly) to _**physical**_ registers (reg. in physical processor) to avoid data hazards. Each _version_ of _architectural_ register is mapped to a unique _physical_ register.

</blockquote>

To make register renaming work, we have a table that maps the _architectural_ registers to the current version of the _physical_ registers. For example:

<div class="small-table">

| Architectural Register | Physical Register |
| ---------------------- | ----------------- |
| `%rax`                 | `%x04`            |
| `%rcx`                 | `%x09`            |
| ...                    | ...               |
| `%r8`                  | `%x13`            |
| `%r9`                  | `%x17`            |
| `%r10`                 | `%x19`            |
| `%r11`                 | `%x07`            |
| `%r12`                 | `%x05`            |

</div>

In the example above, we denote the _architectural_ registers prefix `r` and the _physical_ registers prefix `x`.

We also have a list of phiysical registers that are _**free**_ to use. For example:

<div class="xsmall-table">

| Free Physical Registers |
| :---------------------: |
|         `%x18`          |
|         `%x20`          |
|         `%x21`          |
|         `%x23`          |
|         `%x24`          |
|           ...           |

</div>

We extract a free physical register from the list of free physical registers and map the _architectural_ register to its **new** version of _physical_ register. We also update the list of free physical registers in the `commit` stage of every instruction.

<details open><summary>Register Renaming Example</summary>

Suppose we have the following instructions that follows the mapping above:

1. `add %r10, %r8`
2. `add %r11, %r8`
3. `add %r12, %r8`

We fetch the renamed registers from the register renaming table:

1. `add %r10, %r8` $\implies$ `add %x19, %x13` $\to$ `%x18`

We **always** write to a new physical register from the free register list. We then **update** the register renaming table to map `%r8` to `%x18`.

In the next instruction, the following happens:

2. `add %r11, %r8` $\implies$ `add %x07, %x18` $\to$ `%x20`

We can see that `%r8` uses the new physical register `%x20` in this instruction.

</details>

### **OOO Pipeline Stages**

<img src="https://branyang02.github.io/images/ooo.png" alt="OOO Pipeline" style="display: block; max-height: 70%; max-width: 70%;">

In the OOO pipeline, we have the following stages:

- **Fetch**: fetch the instruction from memory.
- **Decode**: decode the instruction.
- **Rename**: rename the registers, and dispatch the instruction to the _instruction queue_.
- **Issue**: choosing the instructions to execute based on the availability of the execution units, and either read registers from the register file or forward the values from the previous instructions.
- **Execute**: execute the instruction.
- **Writeback**: write the result back to the register file.
- **Commit**: commit the result to the register file, and update the free list of physical registers.

The most important stage in the OOO pipeline is `issue`. This is where we decide which instruction(s) to execute based on the **instruction queue**. The `issue` stage also reads the registers from the register file or finds the forwarded values from the previous instructions.

<blockquote class="important">

`fetch`, `decode`, and `rename` stages are run **in-order**.

`issue`, `execute`, and `writeback` stages are run **out-of-order**.

The final `commit` stage is **in order**.

</blockquote>

<details open><summary>Example OOO Pipeline Stages</summary>

<img src="https://branyang02.github.io/images/ooo_pipeline.png" alt="OOO Pipeline" style="display: block; max-height: 70%; max-width: 70%;">

In the example above, we assume we can execute 2 instructions at once. Therefore, we **must** make sure that we are **not** performing more than _2 stages of the same type_ at the same cycle.

We can see that we `fetch` all the instructions **in order**, followed by `decode` and `rename` stages. However, we can `issue` the instructions **out-of-order**.

For example, `addq %r01, %r05` has `issue` at cycle 3, while the next instruction `addq %r02, %r05` has `issue` at cycle 4. This is because we need to forward the result of the first instruction in its `execute` stage to the `issue` stage of the second instruction.

We can also see that `addq %r02, %r05` and `addq %r03, %r04` have `issue` at the same cycle. This is because they do not have any data dependencies and can be executed in parallel.

Finally, we have the `commit` stage **in order** where we write the result back to the register file.

</details>

### **Instruction Queue and Dispatch Process**

In the `rename` and `issue` stages, we follow the following steps:

1. **Rename**: rename the registers in the instruction to avoid data hazards.
2. **Dispatch**: dispatch the instruction to the **instruction queue**.
3. **Issue**: choose instructions from the instruction queue to execute based on the availability of the execution units.

<img src="https://branyang02.github.io/images/rename_issue.png" alt="Instruction Queue" style="display: block; max-height: 25%; max-width: 25%;">

At the `issue` stage, we need to make sure that we are not issuing more instructions than the number of execution units available. The number of instructions that can be run per cycle is limited by the number of ALU execution units. We can use an **instruction queue** to store the instructions that are ready to be executed.

<details open><summary>Instruction Queue Example</summary>
Continuing from the previous example, suppose we have already converted the instructions to their physical registers:

<div class="small-table">

| #   | Instruction                       |
| --- | --------------------------------- |
| 1   | `addq %x01, %x05` $\to$ `%x06`    |
| 2   | `addq %x02, %x06` $\to$ `%x07`    |
| 3   | `addq %x03, %x07` $\to$ `%x08`    |
| 4   | `cmpq %x04, %x08` $\to$ `%x09.cc` |
| 5   | `jne %x09.cc, ...`                |
| 6   | `addq %x01, %x08` $\to$ `%x10`    |
| 7   | `addq %x02, %x10` $\to$ `%x11`    |
| 8   | `addq %x03, %x11` $\to$ `%x12`    |
| 9   | `cmpq %x04, %x12` $\to$ `%x13.cc` |

</div>
<span class="caption">Example Instruction Queue</span>

We also have a **scoreboard** that keeps track of the status (ready or pending) of the physical registers.

<div class="xsmall-table">

| reg    | status  |
| ------ | ------- |
| `%x01` | ready   |
| `%x02` | ready   |
| `%x03` | ready   |
| `%x04` | ready   |
| `%x05` | ready   |
| `%x06` | pending |
| `%x07` | pending |
| `%x08` | pending |
| `%x09` | pending |
| `%x10` | pending |
| `%x11` | pending |
| `%x12` | pending |
| `%x13` | pending |

</div>
<span class="caption">Example Scoreboard</span>

Suppose we have 2 ALU execution units, meaning we could execute **at most** 2 instructions per cycle. We can use the following steps to dispatch the instructions:

$$

\begin{array}{cccccccccc}
\text{Cycle\#:} & \text{1} & \text{2} & \text{3} & \text{4} & \text{5} & \text{6} & \text{7} \\
\text{ALU 1} & \color{red}1 & \color{red}2 & \color{red}3 & \color{red}4 & \color{red}5 & \color{red}8 & \color{red}9 & & \\
\text{ALU 2} & - & - & - & \color{red}6& \color{red}7 &- & - & & \\
\end{array}


$$

At cycle 1, we can only `issue` instruction `1` to ALU 1 since it is the only instruction that is ready to be executed based on the scoreboard table. After executing instruction `1` at cycle 1, we update the scoreboard table to mark `%x06` as `ready`, and leave `%x05` as ready.

We continue issuing 1 instruction per cycle until we reach cycle 4. At cycle 4, we can `issue` instruction `4` to ALU 1 and instruction `6` to ALU 2 since they are both of the instructions use registers that are ready to be executed based on the scorebaord table.

Refer to the example above to see when `issue` stages occur for each instruction.

</details>

### **Execution Units (Functional Units)**

This is where the `execute` stage of the pipeline occurs. We can have multiple execution units to execute different types of instructions. For example, we can have:

- **ALU**: for arithmetic and logical operations.
- **Pipelined ALU**: for multiple arithmetic and logical operations.
- **Load/Store Unit**: for loading and storing data from memory.

<img src="https://branyang02.github.io/images/execute.png" alt="Execution Units" style="display: block; max-height: 25%; max-width: 25%;">

In the example above, we have `ALU 1` and `ALU 2` that can execute arithmetic and logical operations in 1 cycle. We also have a _pipelined ALU_ that is separated into two parts: `ALU 3 pt1` and `ALU 3 pt2`. This means that a full `ALU 3` operation will take 2 cycles to complete.

<blockquote class="important">

We **only** update the scoreboard table after the entire **ALL** `execute` stages are complete (i.e., if we have 2 ALU unites that take 2 cycles to complete, we only update the scoreboard table after the 2nd cycle).

</blockquote>

The `execute` stage typically _**forwards**_ the result to the `issue` stage of the next instruction.

### **References**

This note is based on [CS 3130 Spring 2024](https://www.cs.virginia.edu/~cr4bd/3130/S2024/) by Charles Reiss, used under CC BY-NC-SA 4.0.
