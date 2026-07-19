/// <reference types="vite/client" />

interface ImportMetaEnv {
    /** Injected at build time via `define` in demo-app/vite.config.ts. */
    readonly PACKAGE_VERSION: string;
}

interface ImportMeta {
    readonly env: ImportMetaEnv;
}
