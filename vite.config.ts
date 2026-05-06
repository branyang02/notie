import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import dts from "vite-plugin-dts";
import cssInjectedByJsPlugin from "vite-plugin-css-injected-by-js";

export default defineConfig({
    test: {
        environment: "jsdom",
        setupFiles: ["src/test/setup.ts"],
        globals: true,
    },
    build: {
        lib: {
            entry: path.resolve(__dirname, "src/index.ts"),
            name: "notie",
            fileName: (format) => `index.${format}.js`,
        },
        rollupOptions: {
            external: ["react", "react-dom"],
            output: {
                globals: {
                    react: "React",
                    "react-dom": "ReactDOM",
                },
            },
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
