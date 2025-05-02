<script lang="ts">
    import { page } from '$app/stores';
</script>

<div class="layout">
    <header>
        <div class="header-content">
            <h1>ElevenLabs MCP Client Example</h1>
            <p>A sample MCP client implementation using SvelteKit, demonstrating integration with ElevenLabs Text-to-Speech API through MCP Server.</p>
        </div>
    </header>
    <nav>
        <a 
            href="/" 
            class="nav-link" 
            class:active={$page.url.pathname === '/'}
        >
            Simple TTS
        </a>
        <a 
            href="/script" 
            class="nav-link" 
            class:active={$page.url.pathname === '/script'}
        >
            Script TTS
        </a>
        <a 
            href="/history" 
            class="nav-link" 
            class:active={$page.url.pathname === '/history'}
        >
            History
        </a>
    </nav>

    <slot />
</div>

<style>
    .layout {
        min-height: 100vh;
        background: var(--color-background);
    }

    header {
        background: var(--color-surface);
        border-bottom: 1px solid var(--border-color);
        padding: var(--spacing-8) var(--spacing-4);
        text-align: center;
    }

    .header-content {
        max-width: 800px;
        margin: 0 auto;
    }

    header h1 {
        color: var(--color-text);
        font-size: var(--font-size-2xl);
        margin: 0 0 var(--spacing-2);
    }

    header p {
        color: var(--color-text-light);
        font-size: var(--font-size-base);
        margin: 0;
        line-height: 1.5;
    }

    nav {
        background: var(--color-surface);
        padding: var(--spacing-4);
        border-bottom: 1px solid var(--border-color);
        display: flex;
        gap: var(--spacing-4);
        justify-content: center;
        box-shadow: var(--shadow-sm);
    }

    .nav-link {
        padding: var(--spacing-2) var(--spacing-4);
        color: var(--color-secondary);
        text-decoration: none;
        border-radius: var(--border-radius-base);
        transition: all var(--transition-base);
        font-size: var(--font-size-base);
    }

    .nav-link:hover {
        background: var(--color-background);
        color: var(--color-text);
        transform: translateY(-1px);
    }

    .nav-link.active {
        background: var(--color-primary);
        color: var(--color-surface);
        font-weight: 500;
    }

    @media (max-width: 640px) {
        header {
            padding: var(--spacing-6) var(--spacing-4);
        }

        header h1 {
            font-size: var(--font-size-xl);
        }

        header p {
            font-size: var(--font-size-sm);
        }

        nav {
            padding: var(--spacing-3);
            gap: var(--spacing-2);
        }

        .nav-link {
            padding: var(--spacing-2) var(--spacing-3);
            font-size: var(--font-size-sm);
        }
    }
</style>
