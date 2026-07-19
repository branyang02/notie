# PR 93 Review Fixture

Intro text before the first section. Literal collision token in prose: NOTIEMASK0NOTIEMASK should render byte-identical.

## See [the guide][guide] here

Reference-style link in a level-2 heading. Body text for the guide section.

### Read [the spec][] now

Collapsed reference link in a level-3 heading.

### Try [shortcut] form

Shortcut reference link.

### Try [shortcut] form

Duplicate heading — must get a `-1` suffix within this section.

[guide]: https://example.com/guide
[the spec]: https://example.com/spec
[shortcut]: https://example.com/shortcut

## Inline [docs](https://example.com/docs) mix

Inline link heading (pre-existing behavior must be preserved).

```bash
## fake heading inside a fence
echo "NOTIEMASK0NOTIEMASK"
```

## Setup

First Setup section.

### Details

Details under first Setup.

## Setup

Duplicate `##` section — rehype-slug resets per section tree, so this must be plain `setup` again, not `setup-1`.

### Details

Duplicate details heading in the second Setup tree.

## Math & C++ (v2)

$$
\begin{equation} \label{eq:collide}
x = 1
\end{equation}
$$

See $\eqref{eq:collide}$.

[guide]: https://example.com/guide
[the spec]: https://example.com/spec
[shortcut]: https://example.com/shortcut
