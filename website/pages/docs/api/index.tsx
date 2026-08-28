import Head from "next/head";
import { DocsLayout, type TocItem } from "@/components/docs/DocsLayout";
import { Endpoint } from "@/components/docs/Endpoint";
import { CodeBlock } from "@/components/docs/CodeBlock";
import { highlightBlocks } from "@/lib/docs/highlight";

const toc: TocItem[] = [
  { id: "get-started", text: "Get started" },
  { id: "operations", text: "Operations" },
  { id: "authentication", text: "Authentication" },
  { id: "errors", text: "Errors" },
];

const BLOCKS = [
  {
    lang: "bash",
    code: `curl -X POST localhost:7240/v1/fork \\
  -H "Content-Type: application/json" \\
  -d '{"image":"ubuntu:24.04"}'`,
  },
  {
    lang: "json",
    code: `{
  "code": 404,
  "message": "VM not found: vm_abc123"
}`,
  },
];

export async function getStaticProps() {
  return { props: { highlighted: await highlightBlocks(BLOCKS) } };
}

// Adapted from arker.ai/docs's API overview (cloudflare/src/content/docs/api/overview.mdx),
// rewritten to describe BLAST's actual scope rather than Arker's: fork, run,
// sync, sessions, and the automatic lifecycle, not multi-tenant orgs,
// billing, GPUs, filesystems, or policies.
export default function ApiOverviewPage({ highlighted }: { highlighted: string[] }) {
  return (
    <>
      <Head>
        <title>API overview | BLAST Documentation</title>
      </Head>
      <DocsLayout toc={toc}>
        <h1>API</h1>
        <p>
          The BLAST API is a small set of operations over VMs: fork one, run
          commands inside it, sync files in and out, and manage sessions.
          It&apos;s a plain REST API served directly by the{" "}
          <code>blast</code> binary, with no separate CLI, web console, or
          client SDK.
        </p>

        <h2 id="get-started">Get started</h2>
        <p>
          Call the API against the port BLAST is listening on (
          <code>7240</code> by default):
        </p>
        <CodeBlock lang="bash" code={BLOCKS[0].code} html={highlighted[0]} />

        <h2 id="operations">Operations</h2>
        <Endpoint
          as="div"
          method="POST"
          path="/v1/fork"
          note="fork a VM"
          href="/docs/api/fork"
        />
        <Endpoint
          as="div"
          method="GET"
          path="/v1/vms"
          note="list VMs"
          href="/docs/api/vms#get-v1-vms"
        />
        <Endpoint
          as="div"
          method="GET"
          path="/v1/vms/{id}"
          note="get a VM"
          href="/docs/api/vms#get-v1-vms-id"
        />
        <Endpoint
          as="div"
          method="DELETE"
          path="/v1/vms/{id}"
          note="delete a VM"
          href="/docs/api/vms#delete-v1-vms-id"
        />
        <Endpoint
          as="div"
          method="POST"
          path="/v1/vms/{id}/runs"
          note="run a command"
          href="/docs/api/runs#post-v1-vms-id-runs"
        />
        <Endpoint
          as="div"
          method="GET"
          path="/v1/vms/{id}/runs/{run_id}"
          note="poll a run"
          href="/docs/api/runs#get-v1-vms-id-runs-run-id"
        />
        <Endpoint
          as="div"
          method="GET"
          path="/v1/vms/{id}/sessions"
          note="list sessions"
          href="/docs/api/sessions#get-v1-vms-id-sessions"
        />
        <Endpoint
          as="div"
          method="POST"
          path="/v1/vms/{id}/sessions"
          note="create a session"
          href="/docs/api/sessions#post-v1-vms-id-sessions"
        />
        <Endpoint
          as="div"
          method="DELETE"
          path="/v1/vms/{id}/sessions/{sid}"
          note="delete a session"
          href="/docs/api/sessions#delete-v1-vms-id-sessions-session-id"
        />
        <Endpoint
          as="div"
          method="POST"
          path="/v1/vms/{id}/sync"
          note="read or write files"
          href="/docs/api/sync"
        />

        <h2 id="authentication">Authentication</h2>
        <p>
          BLAST&apos;s HTTP API has no built-in authentication. Run it behind
          your own network boundary (localhost, a private network, or a
          reverse proxy) if you need to restrict who can call it. The{" "}
          <code>api_key</code> under <code>[worker]</code> in{" "}
          <code>blast.toml</code> is used outbound only, to authenticate
          BLAST to a control plane it registers with; it is not checked on
          inbound requests to BLAST&apos;s own API.
        </p>

        <h2 id="errors">Errors</h2>
        <p>An error response carries the HTTP status and a JSON body:</p>
        <CodeBlock lang="json" code={BLOCKS[1].code} html={highlighted[1]} />
        <p>
          <code>code</code> mirrors the HTTP status; <code>message</code> is
          human-readable.
        </p>
      </DocsLayout>
    </>
  );
}
