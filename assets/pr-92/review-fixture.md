# Review fixture (worktree copy of src/dev/test.md used for visual review)

    # Issue #88 Visual Review

    ## Side by side: classless fence, tagged fence, inline code

    A classless fence (no info string):

    ```
    SELECT * FROM users WHERE active = 1;
    -- no language tag on this fence
    ```

    A language-tagged fence:

    ```python
    def greet(name):
        return f"Hello, {name}!"
    ```

    Inline code stays plain: use `x + 1` and `npm install` inline in a sentence.

    ## Special fences must keep routing

    ```execute-python
    print("live code block")
    ```

    ```component
    {
        componentName: "Widget"
    }
    ```
