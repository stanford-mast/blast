import Head from "next/head";
import { DocsLayout, type TocItem } from "@/components/docs/DocsLayout";
import { Endpoint } from "@/components/docs/Endpoint";
import { Fields, Field } from "@/components/docs/Fields";
import { CodeBlock } from "@/components/docs/CodeBlock";
import { highlightBlocks } from "@/lib/docs/highlight";

const toc: TocItem[] = [
  { id: "the-session-object", text: "The Session object" },
  { id: "post-v1-vms-id-sessions", text: "Create a session" },
  { id: "get-v1-vms-id-sessions", text: "List sessions" },
  { id: "delete-v1-vms-id-sessions-session-id", text: "Delete a session" },
];

const BLOCKS = [
  {
    lang: "bash",
    code: `curl -X POST localhost:7240/v1/vms/$VM/sessions \\
  -H "Content-Type: application/json" \\
  -d '{"cwd":"/app"}'`,
  },
  { lang: "bash", code: `curl localhost:7240/v1/vms/$VM/sessions` },
  {
    lang: "bash",
    code: `curl -X DELETE localhost:7240/v1/vms/$VM/sessions/$SESSION_ID`,
  },
];

export async function getStaticProps() {
  return { props: { highlighted: await highlightBlocks(BLOCKS) } };
}

// Cross-referenced against src/api/types.rs (CreateSessionRequest,
// SessionObject, ListSessionsResponse) and src/api/handlers.rs
// (post_session, list_sessions, delete_session) on the arker-blast host.
export default function ApiSessionsPage({ highlighted }: { highlighted: string[] }) {
  return (
    <>
      <Head>
        <title>Sessions | BLAST Documentation</title>
      </Head>
      <DocsLayout toc={toc}>
        <h1>Sessions</h1>
        <p>
          A session is a persistent shell inside a VM: its own working
          directory, environment, and run queue. Create one to keep state
          across multiple <code>runs</code> calls, the same way a terminal tab
          keeps its own state.
        </p>

        <Endpoint
          as="div"
          method="POST"
          path="/v1/vms/{id}/sessions"
          note="create a session"
          href="#post-v1-vms-id-sessions"
        />
        <Endpoint
          as="div"
          method="GET"
          path="/v1/vms/{id}/sessions"
          note="list sessions"
          href="#get-v1-vms-id-sessions"
        />
        <Endpoint
          as="div"
          method="DELETE"
          path="/v1/vms/{id}/sessions/{session_id}"
          note="delete a session"
          href="#delete-v1-vms-id-sessions-session-id"
        />

        <h2 id="the-session-object">The Session object</h2>
        <Fields>
          <Field name="session_id" type="String">
            The session&apos;s id. Pass this as <code>session_id</code> on{" "}
            <a href="/docs/api/runs" className="link">
              runs
            </a>{" "}
            to execute inside this session.
          </Field>
          <Field name="session_idx" type="Int">
            Zero-based index reflecting creation order within the VM.
          </Field>
          <Field name="state" type='"idle" | "running"'>
            Whether a run is currently executing in this session.
          </Field>
          <Field name="cwd" type="String">
            The session&apos;s working directory.
          </Field>
          <Field name="env" type="Object">
            Environment variables set on this session, if any.
          </Field>
        </Fields>

        <Endpoint
          as="h2"
          method="POST"
          path="/v1/vms/{id}/sessions"
          title="Create a session"
        />
        <h3>Request</h3>
        <Fields>
          <Field name="cwd" type="String">
            Starting working directory for this session.
          </Field>
          <Field name="env" type="Object">
            Environment variables for this session.
          </Field>
        </Fields>
        <h3>Response</h3>
        <p>
          Returns a{" "}
          <a href="#the-session-object" className="link">
            Session object
          </a>
          .
        </p>
        <h3>Example</h3>
        <CodeBlock lang="bash" code={BLOCKS[0].code} html={highlighted[0]} />

        <Endpoint
          as="h2"
          method="GET"
          path="/v1/vms/{id}/sessions"
          title="List sessions"
        />
        <h3>Response</h3>
        <Fields>
          <Field name="sessions" type="Array">
            A list of{" "}
            <a href="#the-session-object" className="link">
              Session objects
            </a>
            .
          </Field>
        </Fields>
        <h3>Example</h3>
        <CodeBlock lang="bash" code={BLOCKS[1].code} html={highlighted[1]} />

        <Endpoint
          as="h2"
          method="DELETE"
          path="/v1/vms/{id}/sessions/{session_id}"
          title="Delete a session"
        />
        <p>Removes the session. A 404 if it doesn&apos;t exist.</p>
        <h3>Example</h3>
        <CodeBlock lang="bash" code={BLOCKS[2].code} html={highlighted[2]} />
      </DocsLayout>
    </>
  );
}
