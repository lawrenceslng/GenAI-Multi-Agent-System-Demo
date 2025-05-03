import { json, type RequestEvent } from "@sveltejs/kit";
import { elevenlabsClient } from "$lib/client";

export async function POST({ request }: RequestEvent) {
  try {
    const { text, voice_id, type = "simple", script } = await request.json();
    if (!elevenlabsClient) {
      throw new Error("MCP client not initialized");
    }

    let result;
    if (type === "simple") {
      result = await elevenlabsClient.generateSimpleAudio(text, voice_id);
    } else if (type === "script") {
      result = await elevenlabsClient.generateScriptAudio(script);
    } else {
      throw new Error(`Invalid TTS type: ${type}`);
    }

    return json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return json(
      {
        success: false,
        message: `Server error: ${message}`,
        debugInfo: [],
      },
      { status: 500 }
    );
  }
}
