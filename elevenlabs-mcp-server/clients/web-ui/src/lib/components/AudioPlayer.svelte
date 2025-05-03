<script lang="ts">
    export let audioData: string; // base64 encoded audio
    export let name: string = 'audio';
    
    let audio: HTMLAudioElement;
    let isPlaying = false;
    
    $: if (audioData) {
        const audioUrl = `data:audio/mpeg;base64,${audioData}`;
        if (audio) {
            audio.src = audioUrl;
        }
    }
    
    function togglePlay() {
        if (audio.paused) {
            audio.play();
        } else {
            audio.pause();
        }
    }
    
    function handlePlayStateChange() {
        isPlaying = !audio.paused;
    }
    
    function download() {
        const link = document.createElement('a');
        link.href = `data:audio/mpeg;base64,${audioData}`;
        link.download = `${name}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
</script>

<div class="audio-player">
    <audio 
        bind:this={audio}
        on:play={handlePlayStateChange}
        on:pause={handlePlayStateChange}
        on:ended={handlePlayStateChange}
    ></audio>
    
    <div class="controls">
        <button 
            class="play-button" 
            on:click={togglePlay} 
            disabled={!audioData}
            aria-label={isPlaying ? 'Pause' : 'Play'}
        >
            {#if isPlaying}
                ⏸️
            {:else}
                ▶️
            {/if}
        </button>
        
        <button 
            class="download-button"
            on:click={download}
            disabled={!audioData}
            aria-label="Download"
        >
            💾
        </button>
    </div>
</div>

<style>
    .audio-player {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        border: 1px solid #ccc;
        border-radius: 0.5rem;
        background: #f9f9f9;
    }
    
    .controls {
        display: flex;
        gap: 0.5rem;
    }
    
    button {
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 0.25rem;
        background: #fff;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    
    button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    button:not(:disabled):hover {
        opacity: 0.8;
    }
</style>
