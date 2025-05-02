import { elevenlabsClient } from "$lib/client";
import { json } from "@sveltejs/kit";

export async function GET() {
  if (!elevenlabsClient) {
    console.error("MCP client not initialized");
    return new Response(
      JSON.stringify({ error: "MCP client not initialized" }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  try {
    console.log("Fetching voice list...");
    const voices = await elevenlabsClient.getVoices();
    console.log("Fetched voices:", voices);
    return json(voices);
  } catch (error) {
    console.error("Error fetching voices:", error);
    return new Response(JSON.stringify({ error: "Failed to fetch voices" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
