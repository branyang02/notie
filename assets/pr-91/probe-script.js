const { chromium } = require("playwright");
const fs = require("fs");

const OUT = "/tmp/notie-visual-review/pr-91";

(async () => {
    const browser = await chromium.launch();
    const ctx = await browser.newContext({
        viewport: { width: 1440, height: 1400 },
    });
    const page = await ctx.newPage();
    const consoleMsgs = [];
    page.on("console", (m) => consoleMsgs.push(`[${m.type()}] ${m.text()}`));

    await page.goto("http://localhost:5485/");
    // Sections reveal progressively; wait for last section content.
    await page.waitForSelector("#section-2", { timeout: 15000 });
    await page.waitForTimeout(2500); // let equation labeler effects settle

    const probe = await page.evaluate(() => {
        const result = { sections: [], eqnNums: [], eqrefs: [], titleSection: {} };

        // All KaTeX-assigned equation numbers in DOM order, with ids.
        document.querySelectorAll(".eqn-num").forEach((el) => {
            result.eqnNums.push({
                id: el.id || null,
                text: (el.textContent || "").trim(),
                inSections: !!el.closest(".sections"),
            });
        });

        // All eqref links: link text should equal the .eqn-num text of the
        // anchor the href points at.
        document.querySelectorAll("a[href^='#eqn-']").forEach((a) => {
            const target = a.getAttribute("href").slice(1);
            const anchor = document.getElementById(target);
            result.eqrefs.push({
                href: a.getAttribute("href"),
                linkText: (a.textContent || "").trim(),
                anchorExists: !!anchor,
                anchorText: anchor ? (anchor.textContent || "").trim() : null,
            });
        });

        // Title section: is there any eqn-num BEFORE the first .sections div
        // (i.e. in the title area), and does any anchor id start with eqn-0?
        result.titleSection.zeroAnchors = Array.from(
            document.querySelectorAll("[id^='eqn-0']"),
        ).map((e) => e.id);
        const firstSection = document.querySelector(".sections");
        let titleEqnNums = 0;
        document.querySelectorAll(".eqn-num").forEach((el) => {
            if (
                firstSection &&
                firstSection.compareDocumentPosition(el) &
                    Node.DOCUMENT_POSITION_PRECEDING
            )
                titleEqnNums++;
        });
        result.titleSection.eqnNumsBeforeFirstSection = titleEqnNums;

        document.querySelectorAll(".sections").forEach((s) => {
            result.sections.push({
                id: s.id,
                eqnNums: Array.from(s.querySelectorAll(".eqn-num")).map(
                    (e) => ({ id: e.id, text: (e.textContent || "").trim() }),
                ),
            });
        });
        return result;
    });

    // Verdict computation: every eqref link text must match its anchor's
    // rendered .eqn-num text, and every anchor must exist.
    const failures = [];
    for (const r of probe.eqrefs) {
        if (!r.anchorExists)
            failures.push(`eqref ${r.href}: anchor MISSING (link text ${r.linkText})`);
        else if (r.linkText !== r.anchorText)
            failures.push(
                `eqref ${r.href}: link text "${r.linkText}" != DOM eqn-num "${r.anchorText}"`,
            );
    }
    if (probe.titleSection.zeroAnchors.length > 0)
        failures.push(
            `found eqn-0.x anchors: ${probe.titleSection.zeroAnchors.join(",")}`,
        );

    probe.failures = failures;
    probe.verdict = failures.length === 0 ? "PASS" : "FAIL";
    probe.consoleErrors = consoleMsgs.filter((m) => m.startsWith("[error]"));

    fs.writeFileSync(`${OUT}/probe.json`, JSON.stringify(probe, null, 2));
    fs.writeFileSync(`${OUT}/console.log`, consoleMsgs.join("\n"));

    await page.screenshot({ path: `${OUT}/full-page.png`, fullPage: true });
    await browser.close();

    console.log(`VERDICT: ${probe.verdict}`);
    console.log(`eqrefs checked: ${probe.eqrefs.length}`);
    failures.forEach((f) => console.log("FAIL: " + f));
})().catch((e) => {
    console.error(e);
    process.exit(1);
});
