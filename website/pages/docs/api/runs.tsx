import Head from "next/head";
import { DocsLayout, type TocItem } from "@/components/docs/DocsLayout";
import { Endpoint } from "@/components/docs/Endpoint";
import { Fields, Field } from "@/components/docs/Fields";
import { CodeBlock } from "@/components/docs/CodeBlock";
import { highlightBlocks } from "@/lib/docs/highlight";

const toc: TocItem[] = [
  { id: "the-run-object", text: "The Run object" },
  { id: "post-v1-vms-id-runs", text: "Run a command" },
  { id: "get-v1-vms-id-runs-run-id", text: "Poll a run" },
];

const BLOCKS = [
  {
    lang: "bash",
    code: `curl -X POST localhost:7240/v1/vms/$VM/runs \\
  -H "Content-Type: application/json" \\
  -d '{"command":"echo hello from fork"}'`,
  },
  { lang: "bash", code: `curl localhost:7240/v1/vms/$VM/runs/$RUN_ID` },
];

export async function getStaticProps() {
  return { props: { highlighted: await highlightBlocks(BLOCKS) } };
}

// Cross-referenced against src/api/types.rs (RunRequest/RunResponse) and
// src/api/handlers.rs (handle_run, get_run) on the arker-blast host.
// `session_idx` is accepted in the request struct but unused by handle_run
// today, so it's left undocumented. There is no list-runs or cancel-run
// endpoint in BLAST's router; only the two documented here.
export default function ApiRunPage({ highlighted }: { highlighted: string[] }) {
  return (
    <>
      <Head>
        <title>Run | BLAST Documentation</title>
      </Head>
      <DocsLayout toc={toc}>
        <h1>Run</h1>
        <p>Run a command in a VM, and poll its result.</p>

        <Endpoint
          as="div"
          method="POST"
          path="/v1/vms/{id}/runs"
          note="run a command"
          href="#post-v1-vms-id-runs"
        />
        <Endpoint
          as="div"
          method="GET"
          path="/v1/vms/{id}/runs/{run_id}"
          note="poll a run"
          href="#get-v1-vms-id-runs-run-id"
        />

        <h2 id="the-run-object">The Run object</h2>
        <Fields>
          <Field name="run_id" type="String">
            The run&apos;s id.
          </Field>
          <Field name="state" type='"running" | "completed" | "failed"'>
            Lifecycle state.
          </Field>
          <Field name="stdout, stderr" type="String">
            The program&apos;s own output, once available.
          </Field>
          <Field
            name="stdout_encoding, stderr_encoding"
            type='"utf-8" | "base64"'
          >
            How <code>stdout</code>/<code>stderr</code> are encoded. Output
            that isn&apos;t valid UTF-8 is sent base64-encoded and tagged
            here.
          </Field>
          <Field name="exit_code" type="Int">
            The command&apos;s exit code, once completed.
          </Field>
          <Field name="fail_reason" type="String">
            Set when <code>state</code> is <code>failed</code>: a
            platform-side explanation, distinct from <code>stderr</code> (the
            program&apos;s own error output).
          </Field>
        </Fields>

        <Endpoint
          as="h2"
          method="POST"
          path="/v1/vms/{id}/runs"
          title="Run a command"
        />
        <h3>Request</h3>
        <Fields>
          <Field name="command" type="String" required>
            The command to run.
          </Field>
          <Field name="session_id" type="String">
            Run inside an existing session (shared shell state). Defaults to
            the VM&apos;s default session.
          </Field>
          <Field name="timeout" type="Int">
            Execution/kill bound in seconds: the longest the command may run
            before it&apos;s killed. Defaults to <code>300</code>.
          </Field>
          <Field name="time_to_background" type="Int">
            Sync window in seconds: how long the HTTP call blocks before
            returning a pollable <code>run_id</code> instead of waiting for
            completion. <code>0</code> returns immediately after the command
            starts. Defaults to <code>300</code>. Does not bound how long the
            command itself runs; that&apos;s <code>timeout</code>.
          </Field>
          <Field name="env" type="Object">
            Environment-variable overrides for this run.
          </Field>
          <Field name="cwd" type="String">
            Working directory for this run. Defaults to <code>/</code>.
          </Field>
        </Fields>

        <h3>Response</h3>
        <p>
          A synchronous run (one that completes inside the sync window)
          returns the completed shape directly:
        </p>
        <Fields>
          <Field name="run_id" type="String">
            The recorded run&apos;s id.
          </Field>
          <Field name="state" type="String">
            <code>completed</code> for this response shape.
          </Field>
          <Field name="stdout, stderr" type="String">
            The program&apos;s own output.
          </Field>
          <Field
            name="stdout_encoding, stderr_encoding"
            type='"utf-8" | "base64"'
          >
            See{" "}
            <a href="#the-run-object" className="link">
              the Run object
            </a>
            .
          </Field>
          <Field name="exit_code" type="Int">
            The command&apos;s exit code.
          </Field>
        </Fields>
        <p>A run that outlives the sync window returns the in-progress shape instead:</p>
        <Fields>
          <Field name="run_id" type="String">
            The id to poll with{" "}
            <a href="#get-v1-vms-id-runs-run-id" className="link">
              GET /v1/vms/{"{id}"}/runs/{"{run_id}"}
            </a>
            .
          </Field>
          <Field name="state" type="String">
            <code>running</code> for this response shape.
          </Field>
        </Fields>
        <h3>Example</h3>
        <CodeBlock lang="bash" code={BLOCKS[0].code} html={highlighted[0]} />

        <Endpoint
          as="h2"
          method="GET"
          path="/v1/vms/{id}/runs/{run_id}"
          title="Poll a run"
        />
        <p>Fetch one run, including its output once it&apos;s available.</p>
        <h3>Response</h3>
        <p>
          Returns a{" "}
          <a href="#the-run-object" className="link">
            Run object
          </a>
          .
        </p>
        <h3>Example</h3>
        <CodeBlock lang="bash" code={BLOCKS[1].code} html={highlighted[1]} />
      </DocsLayout>
    </>
  );
}
