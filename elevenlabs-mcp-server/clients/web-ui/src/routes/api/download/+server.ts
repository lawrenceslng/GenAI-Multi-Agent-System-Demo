import { error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { elevenlabsClient } from "$lib/client";

export const GET: RequestHandler = async ({ url }) => {
  const fileId = url.searchParams.get("id");
  if (!fileId) {
    throw error(400, "Missing file ID");
  }

  if (!elevenlabsClient) {
    throw error(500, "MCP client not initialized");
  }

  try {
    const result = await elevenlabsClient.getAudioFile(fileId);

    if (!result.success || !result.audioData) {
      throw error(404, result.error || "File not found");
    }

    // Convert base64 to Uint8Array
    const binaryString = atob(result.audioData.data);
    const binaryData = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      binaryData[i] = binaryString.charCodeAt(i);
    }

    return new Response(binaryData, {
      headers: {
        "Content-Type": result.audioData.mimeType,
        "Content-Disposition": `attachment; filename="${result.audioData.name}"`,
      },
    });
  } catch (e) {
    console.error("Download error:", e);
    throw error(500, "Failed to download file");
  }
};
