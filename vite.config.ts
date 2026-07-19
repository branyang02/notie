import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import dts from "vite-plugin-dts";
import cssInjectedByJsPlugin from "vite-plugin-css-injected-by-js";

export default defineConfig({
    resolve: {
        alias: {
            // Use the environment-agnostic implementation instead of the
            // browser one (index.dom.js), which calls `document.createElement`
            // at module top level and breaks Node/SSR consumers.
            "decode-named-character-reference": path.resolve(
                __dirname,
                "node_modules/decode-named-character-reference/index.js",
            ),
            "hast-util-from-html-isomorphic": path.resolve(
                __dirname,
                "node_modules/hast-util-from-html-isomorphic/index.js",
            ),
        },
    },
    test: {
        environment: "jsdom",
        setupFiles: ["src/test/setup.ts"],
        globals: true,
    },
    build: {
        lib: {
            entry: path.resolve(__dirname, "src/index.ts"),
            name: "notie",
            formats: ["es", "cjs"],
            fileName: (format) => `index.${format}.js`,
        },
        rollupOptions: {
            external: [/^react($|\/)/, /^react-dom($|\/)/],
        },
        sourcemap: true,
        emptyOutDir: true,
    },
    plugins: [
        react(),
        dts({
            exclude: [
                "src/**/*.test.ts",
                "src/**/*.test.tsx",
                "src/dev/**",
                "src/test/**",
            ],
        }),
        cssInjectedByJsPlugin(),
    ],
    assetsInclude: ["**/*.md"],
});
