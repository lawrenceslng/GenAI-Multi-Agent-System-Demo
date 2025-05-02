<script lang="ts">
    export let info: string[] = [];
    export let expanded: boolean = false;
</script>

<div class="debug-info">
    <button 
        class="toggle"
        on:click={() => expanded = !expanded}
        aria-expanded={expanded}
    >
        Debug Info {expanded ? '▼' : '▶'}
    </button>
    
    {#if expanded}
        <div class="content">
            {#if info.length === 0}
                <p class="empty">No debug information available</p>
            {:else}
                <ul>
                    {#each info as line}
                        <li>{line}</li>
                    {/each}
                </ul>
            {/if}
        </div>
    {/if}
</div>

<style>
    .debug-info {
        margin-top: 1rem;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        overflow: hidden;
    }
    
    .toggle {
        width: 100%;
        padding: 0.5rem;
        background: #f5f5f5;
        border: none;
        text-align: left;
        cursor: pointer;
        font-family: monospace;
    }
    
    .toggle:hover {
        background: #eee;
    }
    
    .content {
        padding: 1rem;
        background: #fafafa;
        font-family: monospace;
        font-size: 0.9rem;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .empty {
        color: #666;
        font-style: italic;
    }
    
    ul {
        margin: 0;
        padding: 0;
        list-style: none;
    }
    
    li {
        padding: 0.25rem 0;
        border-bottom: 1px solid #eee;
    }
    
    li:last-child {
        border-bottom: none;
    }
</style>
