import Head from "next/head";
import { DocsLayout } from "@/components/docs/DocsLayout";

export default function OrchestrationPage() {
  return (
    <>
      <Head>
        <title>Orchestration | BLAST Documentation</title>
      </Head>
      <DocsLayout>
        <h1>Orchestration</h1>
        <p>
          A BLAST server accepts requests to fork and run VMs. If configured
          with a <code>control_plane_endpoint</code>, the BLAST server long
          polls for requests from an external server, relaying information
          necessary for the external control plane to schedule and place VMs
          across one or many BLAST servers. If available, the external
          control plane may coordinate with the BLAST server to move VM
          state between the local machine and durable storage for
          persistence.
        </p>
        <p>
          If you&apos;re building a background coding agent platform that
          requires support for local VMs or VMs on BYOC machines, feel free
          to reach out to us on{" "}
          <a href="https://discord.gg/NqrkJwYYh4" className="link">
            Discord
          </a>{" "}
          or directly contact the developers of BLAST.
        </p>
      </DocsLayout>
    </>
  );
}
