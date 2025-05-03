<script lang="ts">
    export let size: number = 24;
</script>

<div 
    class="spinner"
    style="--size: {size}px"
    role="status"
    aria-label="Loading"
>
    <div class="spinner-inner"></div>
</div>

<style>
    .spinner {
        display: inline-block;
        width: var(--size);
        height: var(--size);
        position: relative;
    }

    .spinner-inner {
        box-sizing: border-box;
        display: block;
        position: absolute;
        width: var(--size);
        height: var(--size);
        border: 2px solid #ccc;
        border-radius: 50%;
        animation: spin 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
        border-color: #ccc transparent transparent transparent;
    }

    @keyframes spin {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }
</style>
