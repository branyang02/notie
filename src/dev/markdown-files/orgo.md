# Example: **Organic Chemistry 2**

<span class="subtitle">
Spring 2024
</span>

## Peptides and Proteins

Peptides and proteins are essential biomolecules composed of amino acids linked by peptide bonds. This section explores the structure, function, and synthesis of peptides and proteins, highlighting their significance in biological systems.

<blockquote class="definition">

- **Peptides**: Short chains of amino acids linked by peptide bonds.
- **Proteins**: Large biomolecules composed of one or more polypeptide chains.
- **Amino Acids**: Building blocks of peptides and proteins, consisting of an amino group, carboxyl group, and side chain.

</blockquote>

```tikz
\usepackage{chemfig}
\begin{document}

\chemfig{[:-90]HN(-[::-45](-[::-45]R)=[::+45]O)>[::+45]*4(-(=O)-N*5(-(<:(=[::-60]O)-[::+60]OH)-(<[::+0])(<:[::-108])-S>)--)}

\end{document}
```

<div class="caption">

Structure of a peptide bond

</div>

## Carbonyl Chemistry

### Overview

Carbonyl compounds are characterized by the presence of a carbonyl group (C=O). This functional group is highly reactive, making carbonyl compounds central in organic synthesis. This section explores the structure, reactivity, and various reactions involving carbonyl groups, including aldehydes, ketones, carboxylic acids, and their derivatives.

### Structure and Properties

The carbonyl group consists of a carbon atom double bonded to an oxygen atom, creating a polar bond due to oxygen's higher electronegativity. This polarization makes the carbonyl carbon electrophilic and the oxygen nucleophilic.

<blockquote class="definition">

- **Electrophilicity of C=O**: The partial positive charge on the carbonyl carbon makes it susceptible to attack by nucleophiles.
- **Resonance Stability**: The carbonyl group exhibits resonance, which contributes to the stability of many carbonyl compounds but also increases reactivity in certain contexts.

</blockquote>

### Key Reactions

#### Nucleophilic Addition

The electrophilic carbon of aldehydes and ketones undergoes nucleophilic addition, which is fundamental to many reactions in carbonyl chemistry.

<blockquote class="example">

1. **Addition of water (Hydration)**:
   - Equation: `RCHO + H2O -> RCH(OH)2`
   - The hydration of aldehydes and ketones forms hydrates, though these are generally reversible and not always isolable under normal conditions.
2. **Addition of alcohols (Acetal Formation)**:
   - Equation: `RCHO + 2ROH -> RCH(OR)2 + H2O`
   - Acetals are formed via the reaction of aldehydes with alcohols, serving as protective groups for carbonyl during synthetic sequences.

</blockquote>

#### Oxidation and Reduction

Aldehydes can be oxidized to carboxylic acids, while ketones are generally resistant to oxidation. Both aldehydes and ketones can be reduced to alcohols.

<blockquote class="example">

- **Oxidation**:
  - Aldehydes to carboxylic acids: `RCHO + [O] -> RCOOH`
- **Reduction**:
  - Ketones to secondary alcohols: `RCOR' + H2 -> RCH(OH)R'`
  - Aldehydes to primary alcohols: `RCHO + H2 -> RCH2OH`

</blockquote>

### Applications in Synthesis

Carbonyl compounds are pivotal in the synthesis of pharmaceuticals, polymers, and perfumes. They serve as intermediates in the synthesis of more complex molecules.

<blockquote class="example">

- **Aldol Condensation**:
  - A powerful reaction that forms a new carbon-carbon bond, essential for building complex molecular structures.
- **Cannizzaro Reaction**:
  - A unique reaction where non-enolizable aldehydes are treated with a strong base to give alcohols and carboxylic acids.

</blockquote>

### Conclusion

Understanding the chemistry of carbonyl compounds is crucial for mastering organic synthesis. Their diverse reactivity patterns enable the construction of a wide range of molecular architectures, making them indispensable in advanced organic synthesis.
