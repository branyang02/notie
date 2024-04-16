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
- [Multitasking](#multitasking)
  - [Processes](#processes)
    - [Context Switch](#context-switch)
    - [Process vs. Thread](#process-vs-thread)
- [Signals](#signals)
  - [Signal vs. Exception](#signal-vs-exception)
  - [Forwarding exceptions to signals](#forwarding-exceptions-to-signals)
  - [Common Signals](#common-signals)
  - [Signals Setup](#signals-setup)
  - [Handling multiple signals](#handling-multiple-signals)
  - [Blocking Signals](#blocking-signals)
- [References](#references)
</details>

#### **Introduction**

These are my notes for Computer Systems and Organization 2 (CSO2) at the University of Virginia in the Spring 2024 semester taught by Charles Reiss. This note contains live code examples and explanations for various topics in the course.

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

###### **Rules**:

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

| Exceptions     | Classify by Cause                                                 | Classify by Result                                         |
| -------------- | ----------------------------------------------------------------- | ---------------------------------------------------------- |
| **Interrupts** | occurs independently from the code being executed when it occurs. | runs each instruction once (has no triggering instruction) |
| **Faults**     | An instruction failing to suceed in its execution.                | re-runs triggering instruction                             |
| **Traps**      | caused by a special instruction whose purpose is to cause a trap  | runs instruction after triggering instruction              |

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

##### **System Calls**

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

#### **Multitasking**

- **Multitasking**: a generic term for having multiple processes running on a single machine.
- **Preemptive Multitasking**: the operating system can interrupt a process and switch to another process.
- **Cooperative Multitasking**: the process must voluntarily give up control.

##### **Processes**

- **Process**: an instance of a program in execution, acts like a _virtual machine_.
  - A process has its own program registers, condition codes, **virtual address space**, etc.
- **Virtual Address Space**: the memory that a process can access. (illusion of a program having its own memory)

<img src="https://branyang02.github.io/images/address_space.png" alt="Virtual Address Space" style="display: block; max-height: 70%; max-width: 70%;">
<span
    class="caption"> The virtual address space is the memory that a process can access. It is an illusion of a program having its own memory.
</span>

###### **Context Switch**

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

###### **Process** vs. **Thread**:

- **Process**: an instance of a program in execution.
- **Thread**: a process can have multiple threads of execution. Threads share the same **virtual address space**, but have their own **program registers**, **program counter**, condition codes, etc.

  <div style="background-color: white;">
      <img src="https://static.javatpoint.com/difference/images/process-vs-thread3.png" style="max-height: 70%; max-width: 70%;">
  </div>
  <span
      class="caption"> Threads within the same process share the same virtual address space but have their own program registers, program counter, condition codes, etc. (Source: javapoint, <a href="https://www.javatpoint.com/process-vs-thread">Process Vs. Thread</a>)
  </span>

#### **Signals**

- **Signal**: a way to notify a process that an event has occurred.
- **Signal Handler**: a function that is called when a signal is received.
  - Ex. `SIGINT` (interrupt from keyboard), `SIGSEGV` (segmentation fault), `SIGKILL` (kill the process).

##### **Signal vs. Exception**

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

##### **Forwarding exceptions to signals**

<img src="https://branyang02.github.io/images/signals.png" alt="Signals" style="display: block; max-height: 70%; max-width: 70%;">
<span
      class="caption"> When `SIGINT` is received, the program enters kernel mode and starts running the exception handler for handing keyboard interrupts. The exception handler then forwards the signal to the user mode signal handler. The signal handler then runs in user mode. After the signal handler finishes, the program enters the kernel mode again to clean up and return to user mode.
</span>

##### **Common Signals**

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

##### **Signals Setup**

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

##### **Handling multiple signals**

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

##### **Blocking Signals**

- Use `sigprocmask()` to block signals.

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
- `sigwait()` blocks until a signal is received.

#### **Processes**

coming soon...

#### **Virtual Memory**

coming soon...

#### **Cache**

coming soon...

#### **Synchonization**

coming soon...

#### **Networking**

coming soon...

#### **Secure Channels**

Secure communication over an insecure channel involves safeguarding the data against unauthorized access and manipulation. The two primary concerns are confidentiality and authenticity, ensuring the data remains private and verifiable respectively.

##### **Example Attack Scenarios**

- **Passive Attacks:**
  - **Eavesdropping:** An unauthorized party listens to the communications between Machine A and Machine B.
- **Active Attacks:**
  - **Machine-in-the-middle Attack:** An attacker intercepts, alters, or injects messages between Machine A and Machine B.

<img src="https://images.shiksha.com/mediadata/ugcDocuments/images/wordpressImages/2023_02_MicrosoftTeams-image-304.jpg" alt="Secure Channels" style="display: block; max-height: 70%; max-width: 70%;">
<span
    class="caption"> Source: <a href="https://www.shiksha.com/online-courses/articles/difference-between-active-and-passive-attacks/">Difference Between Active and Passive Attacks</a>

</span>

##### **Confidentiality**

- Confidentiality ensures that the message is only readable by the intended recipient.

**Example Scenario:**

- Machine A sends a message to Machine B.
- Machine M, acting as a machine-in-the-middle, intercepts and may alter the message pretending to be Machine B.

###### **Symmetric Encryption Functions**

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
```

###### **Asymmetric Encryption Functions**

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

##### **Authenticity**

- Authenticity ensures the message is actually sent by the claimed sender.

###### **Message Authentication Code (MAC)**

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

###### **Digital Signatures**

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

##### **Handling Replay Attacks**

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

##### **Certificate**

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

### **References**

This note is based on [CS 3130 Spring 2024](https://www.cs.virginia.edu/~cr4bd/3130/S2024/) by Charles Reiss, used under CC BY-NC-SA 4.0.
