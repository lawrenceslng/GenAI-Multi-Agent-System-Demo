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
    console.log("Fetching job history...");
    const jobs = await elevenlabsClient.getJobHistory();
    console.log("Fetched jobs:", jobs);
    return json(jobs);
  } catch (error) {
    console.error("Error fetching history:", error);
    return new Response(JSON.stringify({ error: "Failed to fetch history" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}

export async function DELETE({ url }) {
  if (!elevenlabsClient) {
    return new Response(
      JSON.stringify({ error: "MCP client not initialized" }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }

  const jobId = url.searchParams.get("id");
  if (!jobId) {
    return new Response(JSON.stringify({ error: "Job ID is required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  try {
    const success = await elevenlabsClient.deleteJob(jobId);
    if (success) {
      return json({ success: true });
    } else {
      return new Response(JSON.stringify({ error: "Failed to delete job" }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      });
    }
  } catch (error) {
    console.error("Error deleting job:", error);
    return new Response(JSON.stringify({ error: "Failed to delete job" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
