<script lang="ts">
    import AudioPlayer from '$lib/components/AudioPlayer.svelte';
    import DebugInfo from '$lib/components/DebugInfo.svelte';
    import LoadingSpinner from '$lib/components/LoadingSpinner.svelte';
    import type { AudioGenerationResponse, Voice } from '$lib/elevenlabs-client';
    import { onMount } from 'svelte';

    let text = '';
    let voiceDetailsExpanded = false;
    let voiceId = 'dQn9HIMKSXWzKBGkbhfP'; // Default to Atom-Pro
    let loading = false;
    let result: AudioGenerationResponse | null = null;
    let voices: Voice[] = [];

    onMount(async () => {
        try {
            const response = await fetch('/api/voices');
            voices = await response.json();
        } catch (error) {
            console.error('Error loading voices:', error);
        }
    });

    async function generateAudio() {
        if (!text) return;
        
        loading = true;
        result = null;
        
        try {
            const response = await fetch('/api/tts', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text,
                    voice_id: voiceId || undefined,
                    type: 'simple'
                })
            });
            
            const data = await response.json() as AudioGenerationResponse;
            
            if (!data.success) {
                throw new Error(data.message);
            }
            
            result = data;
        } catch (error) {
            result = {
                success: false,
                message: error instanceof Error ? error.message : String(error),
                debugInfo: []
            };
        } finally {
            loading = false;
        }
    }

    const formatLabels = (voice: Voice) =>{
        if (!voice.labels) return 'No labels';
        return Object.entries(voice.labels).map(([key, value]) => `${key}: ${value}`).join(', ');
    }

    function getSelectedVoice() {
        return voices.find(v => v.voice_id === voiceId);
    }
</script>

<main>
    <h2>Basic Text-to-Speech Conversion</h2>
    <p class="page-description">Convert single text input to speech using optional voice ID.</p>
    
    <form on:submit|preventDefault={generateAudio} class="tts-form">
        <div class="form-group">
            <label for="text">Text</label>
            <textarea
                id="text"
                bind:value={text}
                placeholder="Enter text to convert to speech..."
                rows="4"
                required
            ></textarea>
        </div>
        
        <div class="form-group">
            <label for="voice">Voice</label>
            <select
                id="voice"
                bind:value={voiceId}
                required
            >
                {#each voices as voice}
                    <option value={voice.voice_id}>
                        {voice.name} ({voice.category})
                    </option>
                {/each}
            </select>
        </div>
        
        <button type="submit" disabled={loading || !text}>
            {#if loading}
                <LoadingSpinner size={16} />
                Generating...
            {:else}
                Generate Audio
            {/if}
        </button>
    </form>
    
    {#if getSelectedVoice()}
        {@const voice = getSelectedVoice()}
        <div class="voice-details">
            <button 
                type="button" 
                class="toggle-details"
                on:click={() => voiceDetailsExpanded = !voiceDetailsExpanded}
                aria-expanded={voiceDetailsExpanded}
            >
                <h3>Selected Voice Details</h3>
                <span class="toggle-icon">{voiceDetailsExpanded ? '▼' : '▶'}</span>
            </button>
            {#if voiceDetailsExpanded}
            <div class="voice-info">
                <div class="info-group">
                    <span class="label">Name:</span>
                    <span>{voice ? voice.name : 'Unknown'}</span>
                </div>
                {#if voice && voice.description}
                    <div class="info-group">
                        <span class="label">Description:</span>
                        <span>{voice.description}</span>
                    </div>
                {/if}
                {#if voice && voice.labels}
                    <div class="info-group">
                        <span class="label">Labels:</span>
                        <span>{formatLabels(voice)}</span>
                    </div>
                {/if}
                {#if voice && voice.preview_url}
                    <div class="preview-audio">
                        <span class="label">Preview:</span>
                        <audio controls src={voice.preview_url}>
                            <track kind="captions">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                {/if}
            </div>
            {/if}
        </div>
    {/if}

    {#if result}
        <div class="result">
            {#if result.success && result.audioData}
                <AudioPlayer 
                    audioData={result.audioData.data}
                    name={result.audioData.name}
                />
            {:else}
                <p class="error">{result.message}</p>
            {/if}
            
            <DebugInfo info={result.debugInfo} />
        </div>
    {/if}
</main>

<style>
    main {
        max-width: 800px;
        margin: 0 auto;
        padding: var(--spacing-8);
    }
    
    h2 {
        margin-bottom: var(--spacing-2);
        color: var(--color-text);
        font-size: var(--font-size-2xl);
        text-align: center;
    }

    .page-description {
        text-align: center;
        color: var(--color-text-light);
        margin-bottom: var(--spacing-8);
    }
    
    .tts-form {
        display: flex;
        flex-direction: column;
        gap: var(--spacing-6);
        margin-bottom: var(--spacing-8);
        background: var(--color-surface);
        padding: var(--spacing-6);
        border-radius: var(--border-radius-lg);
        box-shadow: var(--shadow-base);
    }
    
    .form-group {
        display: flex;
        flex-direction: column;
        gap: var(--spacing-2);
    }
    
    label {
        font-weight: 500;
        color: var(--color-text);
        font-size: var(--font-size-sm);
    }
    
    textarea, input, select {
        padding: var(--spacing-3);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-base);
        font-size: var(--font-size-base);
        background: var(--color-background);
        transition: all var(--transition-base);
    }
    
    textarea:focus, input:focus, select:focus {
        outline: none;
        border-color: var(--color-primary);
        box-shadow: var(--shadow-sm);
    }
    
    button {
        padding: var(--spacing-3) var(--spacing-6);
        background: var(--color-primary);
        color: var(--color-surface);
        border: none;
        border-radius: var(--border-radius-base);
        font-size: var(--font-size-base);
        font-weight: 500;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: var(--spacing-2);
        transition: all var(--transition-base);
        box-shadow: var(--shadow-sm);
    }
    
    button:disabled {
        opacity: 0.7;
        cursor: not-allowed;
        transform: none;
    }
    
    button:not(:disabled):not(.toggle-details):hover {
        background: var(--color-primary-dark);
        transform: translateY(-1px);
        box-shadow: var(--shadow-base);
    }
    
    .result {
        margin-top: var(--spacing-8);
        background: var(--color-surface);
        padding: var(--spacing-6);
        border-radius: var(--border-radius-lg);
        box-shadow: var(--shadow-base);
    }
    
    .error {
        color: var(--color-error);
        padding: var(--spacing-4);
        background: #fef2f2;
        border: 1px solid #fee2e2;
        border-radius: var(--border-radius-base);
        margin-bottom: var(--spacing-4);
    }

    .voice-details {
        margin-top: var(--spacing-8);
        background: var(--color-surface);
        padding: var(--spacing-6);
        border-radius: var(--border-radius-lg);
        box-shadow: var(--shadow-base);
    }

    .toggle-details {
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: none;
        border: none;
        padding: 0;
        margin-bottom: var(--spacing-3);
        cursor: pointer;
        color: var(--color-text);
    }


    .toggle-details h3 {
        font-size: var(--font-size-lg);
        margin: 0;
        color: inherit;
    }

    .toggle-icon {
        font-size: var(--font-size-lg);
        font-weight: bold;
        color: var(--color-text-light);
    }

    .voice-info {
        display: flex;
        flex-direction: column;
        gap: var(--spacing-3);
    }

    .info-group {
        display: flex;
        flex-direction: column;
        gap: var(--spacing-1);
    }

    .preview-audio {
        display: flex;
        flex-direction: column;
        gap: var(--spacing-2);
        margin-top: var(--spacing-2);
    }

    .preview-audio audio {
        width: 100%;
        margin-top: var(--spacing-1);
    }

    .label {
        font-weight: 500;
        color: var(--color-text-light);
        font-size: var(--font-size-sm);
    }

    @media (max-width: 640px) {
        main {
            padding: var(--spacing-4);
        }

        h2 {
            font-size: var(--font-size-xl);
            margin-bottom: var(--spacing-2);
        }

        .page-description {
            font-size: var(--font-size-sm);
            margin-bottom: var(--spacing-6);
        }

        .tts-form {
            padding: var(--spacing-4);
            gap: var(--spacing-4);
        }

        .result {
            padding: var(--spacing-4);
        }

        .voice-details {
            padding: var(--spacing-4);
            margin-top: var(--spacing-6);
        }

        .voice-details h3 {
            font-size: var(--font-size-base);
            margin-bottom: var(--spacing-3);
        }
    }
</style>
