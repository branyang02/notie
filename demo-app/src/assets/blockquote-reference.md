# Example: Blockquote References

The source code for this markdown file can be found in [blockquote-reference.md](https://github.com/branyang02/notie/blob/main/demo-app/src/assets/blockquote-reference.md).

## Overview

**notie** supports hover-preview references for labeled blockquotes. Supported types are:
`definition`, `theorem`, `lemma`, `algorithm`, and `problem`.

## Defining a Labeled Blockquote

Add an `id` attribute to any supported blockquote type:

```html
<blockquote class="definition" id="def:quadratic">
  A **quadratic equation** is a polynomial equation of degree 2 of the form
  $ax^2 + bx + c = 0$, where $a \neq 0$.
</blockquote>
```

<blockquote class="definition" id="def:quadratic">

A **quadratic equation** is a polynomial equation of degree 2 of the form $ax^2 + bx + c = 0$, where $a \neq 0$.

</blockquote>

## Referencing a Blockquote

Use a Markdown link with the `#bqref-` prefix followed by the blockquote's `id`:

```markdown
See [Definition](#bqref-def:quadratic) for the definition of a quadratic equation.
```

See [Definition](#bqref-def:quadratic) for the definition of a quadratic equation. Hovering over the link shows a preview of the blockquote content. Clicking navigates to the blockquote itself.

## Theorems and Lemmas

Theorems and lemmas **share a counter** within each section, matching standard academic numbering convention.

<blockquote class="theorem" id="thm:quadratic-formula">

The solutions to a quadratic equation $ax^2 + bx + c = 0$ are given by:

$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

</blockquote>

<blockquote class="lemma" id="lem:discriminant">

The **discriminant** $\Delta = b^2 - 4ac$ determines the nature of the roots: if $\Delta > 0$ there are two real roots, if $\Delta = 0$ there is one repeated root, and if $\Delta < 0$ there are two complex roots.

</blockquote>

[Theorem](#bqref-thm:quadratic-formula) gives the quadratic formula. [Lemma](#bqref-lem:discriminant) characterizes the roots based on the discriminant defined in [Definition](#bqref-def:quadratic).

## Algorithms and Problems

<blockquote class="algorithm" id="alg:binary-search">

**Input:** Sorted array $A$, target value $t$

1. Set $lo = 0$, $hi = |A| - 1$
2. While $lo \leq hi$: let $mid = \lfloor (lo + hi) / 2 \rfloor$; if $A[mid] = t$ return $mid$; if $A[mid] < t$ set $lo = mid + 1$; else set $hi = mid - 1$
3. Return $-1$

</blockquote>

<blockquote class="problem" id="prob:sorting">

Given an unsorted array of $n$ integers, sort them in $O(n \log n)$ time.

</blockquote>

[Algorithm](#bqref-alg:binary-search) runs in $O(\log n)$ time and assumes the input satisfies [Problem](#bqref-prob:sorting).

## Disabling Previews

Set `previewBlockquotes: false` in the config to render plain links without tooltips:

```tsx
<Notie markdown={markdown} config={{ previewBlockquotes: false }} />
```
