<script lang="ts">
    import AudioPlayer from '$lib/components/AudioPlayer.svelte';
    import DebugInfo from '$lib/components/DebugInfo.svelte';
    import LoadingSpinner from '$lib/components/LoadingSpinner.svelte';
    import type { AudioGenerationResponse, ScriptPart, Voice } from '$lib/elevenlabs-client';
    import { onMount } from 'svelte';

    let scriptParts: ScriptPart[] = [{ text: '', voice_id: 'dQn9HIMKSXWzKBGkbhfP', actor: '' }];
    let expandedVoiceDetails: { [key: number]: boolean } = {};
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

    function addPart() {
        scriptParts = [...scriptParts, { text: '', voice_id: '', actor: '' }];
    }

    function removePart(index: number) {
        scriptParts = scriptParts.filter((_, i) => i !== index);
    }

    function getSelectedVoice(voiceId: string) {
        return voices.find(v => v.voice_id === voiceId);
    }
    
    $: isFormInvalid = scriptParts.length === 0 || scriptParts.some(part => !part.text || !part.voice_id);
    
    async function generateAudio() {
        if (scriptParts.length === 0 || scriptParts.every(part => !part.text)) return;
        
        loading = true;
        result = null;
        
        try {
            const response = await fetch('/api/tts', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    type: 'script',
                    script: { script: scriptParts }
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
</script>

<main>
    <h2>Multi-part Script to Speech</h2>
    <p class="page-description">Create a script with multiple parts, each with its own text, voice, and actor.</p>
    
    <form on:submit|preventDefault={generateAudio} class="script-form">
        {#each scriptParts as part, i}
            <div class="script-part">
                <div class="part-header">
                    <h3>Part {i + 1}</h3>
                    {#if scriptParts.length > 1}
                        <button 
                            type="button" 
                            class="remove-button"
                            on:click={() => removePart(i)}
                            aria-label="Remove part"
                        >
                            ✕
                        </button>
                    {/if}
                </div>
                
                <div class="form-group">
                    <label for="text-{i}">Text</label>
                    <textarea
                        id="text-{i}"
                        bind:value={part.text}
                        placeholder="Enter text for this part..."
                        rows="3"
                        required
                    ></textarea>
                </div>
                
                <div class="part-details">
                    <div class="form-group">
                        <label for="voice-{i}">Voice</label>
                        <select
                            id="voice-{i}"
                            bind:value={part.voice_id}
                            required
                        >
                            {#each voices as voice}
                                <option value={voice.voice_id}>
                                    {voice.name} ({voice.category})
                                </option>
                            {/each}
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="actor-{i}">Actor Name</label>
                        <input
                            id="actor-{i}"
                            type="text"
                            bind:value={part.actor}
                            placeholder="Enter actor name..."
                        />
                    </div>
                </div>

                {#if part.voice_id && getSelectedVoice(part.voice_id)}
                    {@const voice = getSelectedVoice(part.voice_id)}
                    <div class="voice-details">
                        <button 
                            type="button" 
                            class="toggle-details"
                            on:click={() => expandedVoiceDetails[i] = !expandedVoiceDetails[i]}
                            aria-expanded={expandedVoiceDetails[i]}
                        >
                            <h4>Voice Details</h4>
                            <span class="toggle-icon">{expandedVoiceDetails[i] ? '▼' : '▶'}</span>
                        </button>
                        {#if expandedVoiceDetails[i]}
                        <div class="voice-info">
                            <div class="info-group">
                                <span class="label">Name:</span>
                                <span>{voice ? voice.name : 'Unknown Voice'}</span>
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
                                    <span>{Object.entries(voice.labels)
                                        .map(([key, value]) => `${key}: ${value}`)
                                        .join(', ')}</span>
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
            </div>
        {/each}
        
        <div class="form-actions">
            <button 
                type="button" 
                class="secondary-button"
                on:click={addPart}
            >
                Add Part
            </button>
            
            <button 
                type="submit" 
                class="primary-button"
                disabled={loading || isFormInvalid}
            >
                {#if loading}
                    <LoadingSpinner size={16} />
                    Generating...
                {:else}
                    Generate Audio
                {/if}
            </button>
        </div>
    </form>
    
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
    
    .script-form {
        display: flex;
        flex-direction: column;
        gap: var(--spacing-6);
        margin-bottom: var(--spacing-8);
    }
    
    .script-part {
        padding: var(--spacing-6);
        background: var(--color-surface);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius-lg);
        box-shadow: var(--shadow-base);
    }
    
    .part-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--spacing-4);
    }
    
    h3 {
        margin: 0;
        color: var(--color-text);
        font-size: var(--font-size-lg);
    }
    
    .remove-button {
        padding: var(--spacing-1) var(--spacing-2);
        background: var(--color-error);
        color: white;
        border: none;
        border-radius: var(--border-radius-sm);
        cursor: pointer;
        font-size: var(--font-size-sm);
        transition: all var(--transition-base);
    }
    
    .remove-button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    
    .form-group {
        display: flex;
        flex-direction: column;
        gap: var(--spacing-2);
        margin-bottom: var(--spacing-4);
    }

    .voice-details {
        margin-top: var(--spacing-4);
        background: var(--color-surface-light);
        padding: var(--spacing-4);
        border-radius: var(--border-radius-base);
        border: 1px solid var(--border-color);
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

    .toggle-details:hover {
        opacity: 0.8;
    }

    .toggle-details h4 {
        font-size: var(--font-size-base);
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
    
    .part-details {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--spacing-4);
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
    
    .form-actions {
        display: flex;
        gap: var(--spacing-4);
        justify-content: flex-end;
        margin-top: var(--spacing-4);
    }
    
    button {
        padding: var(--spacing-3) var(--spacing-6);
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
    
    .primary-button {
        background: var(--color-primary);
        color: var(--color-surface);
    }
    
    .secondary-button {
        background: var(--color-secondary-light);
        color: var(--color-text);
    }
    
    button:disabled {
        opacity: 0.7;
        cursor: not-allowed;
        transform: none;
    }
    
    button:not(:disabled):hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-base);
    }

    .primary-button:not(:disabled):hover {
        background: var(--color-primary-dark);
    }

    .secondary-button:not(:disabled):hover {
        background: var(--color-secondary);
        color: var(--color-surface);
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

    @media (max-width: 640px) {
        main {
            padding: var(--spacing-4);
        }

        .voice-details {
            padding: var(--spacing-3);
            margin-top: var(--spacing-3);
        }

        .voice-details h4 {
            font-size: var(--font-size-sm);
            margin-bottom: var(--spacing-2);
        }

        .info-group {
            font-size: var(--font-size-sm);
        }

        h2 {
            font-size: var(--font-size-xl);
            margin-bottom: var(--spacing-2);
        }

        .page-description {
            font-size: var(--font-size-sm);
            margin-bottom: var(--spacing-6);
        }

        .script-part {
            padding: var(--spacing-4);
        }

        .part-details {
            grid-template-columns: 1fr;
        }

        .form-actions {
            flex-direction: column;
        }

        .form-actions button {
            width: 100%;
        }

        .result {
            padding: var(--spacing-4);
        }
    }
</style>
