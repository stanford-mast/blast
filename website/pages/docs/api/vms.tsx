import Head from "next/head";
import { DocsLayout, type TocItem } from "@/components/docs/DocsLayout";
import { Endpoint } from "@/components/docs/Endpoint";
import { Fields, Field } from "@/components/docs/Fields";
import { CodeBlock } from "@/components/docs/CodeBlock";
import { highlightBlocks } from "@/lib/docs/highlight";

const toc: TocItem[] = [
  { id: "the-vm-object", text: "The VM object" },
  { id: "the-session-object", text: "The Session object" },
  { id: "get-v1-vms", text: "List VMs" },
  { id: "get-v1-vms-id", text: "Get VM" },
  { id: "delete-v1-vms-id", text: "Delete VM" },
  { id: "get-v1-vms-id-sessions", text: "List sessions" },
  { id: "post-v1-vms-id-sessions", text: "Create session" },
  { id: "delete-v1-vms-id-sessions-sid", text: "Delete session" },
];

const BLOCKS = [
  { lang: "bash", code: `curl localhost:7240/v1/vms` },
  { lang: "bash", code: `curl localhost:7240/v1/vms/$VM` },
  { lang: "bash", code: `curl -X DELETE localhost:7240/v1/vms/$VM` },
  { lang: "bash", code: `curl localhost:7240/v1/vms/$VM/sessions` },
  {
    lang: "bash",
    code: `curl -X POST localhost:7240/v1/vms/$VM/sessions \\
  -H "Content-Type: application/json" \\
  -d '{"cwd":"/tmp"}'`,
  },
  {
    lang: "bash",
    code: `curl -X DELETE localhost:7240/v1/vms/$VM/sessions/$SID`,
  },
];

export async function getStaticProps() {
  return { props: { highlighted: await highlightBlocks(BLOCKS) } };
}

// Cross-referenced against src/api/types.rs (VmObject, SessionObject) and
// src/api/handlers.rs on the arker-blast host. There is no GET-by-id endpoint
// for a single session (only list/create/delete) and no query filtering on
// GET /v1/vms, both match the real router exactly, not arker.ai's fuller set.
export default function ApiVmsPage({ highlighted }: { highlighted: string[] }) {
  return (
    <>
      <Head>
        <title>VMs | BLAST Documentation</title>
      </Head>
      <DocsLayout toc={toc}>
        <h1>VMs</h1>
        <p>List, inspect, and delete VMs, and manage their sessions.</p>

        <Endpoint
          as="div"
          method="GET"
          path="/v1/vms"
          note="list VMs"
          href="#get-v1-vms"
        />
        <Endpoint
          as="div"
          method="GET"
          path="/v1/vms/{id}"
          note="get a VM"
          href="#get-v1-vms-id"
        />
        <Endpoint
          as="div"
          method="DELETE"
          path="/v1/vms/{id}"
          note="delete a VM"
          href="#delete-v1-vms-id"
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
          method="POST"
          path="/v1/vms/{id}/sessions"
          note="create a session"
          href="#post-v1-vms-id-sessions"
        />
        <Endpoint
          as="div"
          method="DELETE"
          path="/v1/vms/{id}/sessions/{sid}"
          note="delete a session"
          href="#delete-v1-vms-id-sessions-sid"
        />

        <h2 id="the-vm-object">The VM object</h2>
        <Fields>
          <Field name="vm_id" type="String">
            The VM&apos;s id.
          </Field>
          <Field name="name" type="String | null">
            The VM&apos;s name, if one was given at fork time.
          </Field>
          <Field name="state" type='"running" | "idle"'>
            Lifecycle state. A run against an idle VM auto-resumes it.
          </Field>
          <Field name="provider, region" type="String">
            Where this VM runs, from the worker&apos;s own{" "}
            <code>[worker]</code> config.
          </Field>
          <Field name="platform" type="String">
            The backend&apos;s platform identifier for this VM.
          </Field>
          <Field name="resources" type="Object">
            Provisioned resources: <code>vcpu</code>, <code>memory_mib</code>,{" "}
            <code>disk_mib</code>.
          </Field>
        </Fields>

        <h2 id="the-session-object">The Session object</h2>
        <p>
          A session is a persistent shell whose working directory and
          environment survive across runs. A VM gets a default session the
          first time it runs a command without an explicit{" "}
          <code>session_id</code>; open more with{" "}
          <a href="#post-v1-vms-id-sessions" className="link">
            create session
          </a>
          .
        </p>
        <Fields>
          <Field name="session_id" type="String">
            The session&apos;s id.
          </Field>
          <Field name="session_idx" type="Int">
            Its index on the VM; the first session created is{" "}
            <code>0</code>.
          </Field>
          <Field name="state" type='"idle" | "running"'>
            Lifecycle state.
          </Field>
          <Field name="cwd" type="String">
            Current working directory.
          </Field>
          <Field name="env" type="Object | null">
            Environment-variable overrides for the session, if any were set.
          </Field>
        </Fields>

        <Endpoint as="h2" method="GET" path="/v1/vms" title="List VMs" />
        <p>List every VM known to this worker.</p>
        <h3>Response</h3>
        <Fields>
          <Field name="vms" type="Array">
            An array of{" "}
            <a href="#the-vm-object" className="link">
              VM objects
            </a>
            .
          </Field>
        </Fields>
        <h3>Example</h3>
        <CodeBlock lang="bash" code={BLOCKS[0].code} html={highlighted[0]} />

        <Endpoint as="h2" method="GET" path="/v1/vms/{id}" title="Get VM" />
        <p>Fetch a single VM.</p>
        <h3>Response</h3>
        <p>
          Returns a{" "}
          <a href="#the-vm-object" className="link">
            VM object
          </a>
          .
        </p>
        <h3>Example</h3>
        <CodeBlock lang="bash" code={BLOCKS[1].code} html={highlighted[1]} />

        <Endpoint
          as="h2"
          method="DELETE"
          path="/v1/vms/{id}"
          title="Delete VM"
        />
        <p>Delete a VM and release its resources.</p>
        <h3>Response</h3>
        <Fields>
          <Field name="deleted" type="Bool">
            <code>true</code> once the VM is gone.
          </Field>
        </Fields>
        <h3>Example</h3>
        <CodeBlock lang="bash" code={BLOCKS[2].code} html={highlighted[2]} />

        <Endpoint
          as="h2"
          method="GET"
          path="/v1/vms/{id}/sessions"
          title="List sessions"
        />
        <p>List a VM&apos;s sessions.</p>
        <h3>Response</h3>
        <Fields>
          <Field name="sessions" type="Array">
            An array of{" "}
            <a href="#the-session-object" className="link">
              Session objects
            </a>
            .
          </Field>
          <Field name="next_cursor" type="null">
            Always <code>null</code> today; session listing is not
            paginated.
          </Field>
        </Fields>
        <h3>Example</h3>
        <CodeBlock lang="bash" code={BLOCKS[3].code} html={highlighted[3]} />

        <Endpoint
          as="h2"
          method="POST"
          path="/v1/vms/{id}/sessions"
          title="Create session"
        />
        <p>Open a new session on a VM.</p>
        <h3>Request</h3>
        <Fields>
          <Field name="cwd" type="String">
            Starting working directory. Defaults to <code>/</code>.
          </Field>
          <Field name="env" type="Object">
            Environment-variable overrides for the session.
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
        <CodeBlock lang="bash" code={BLOCKS[4].code} html={highlighted[4]} />

        <Endpoint
          as="h2"
          method="DELETE"
          path="/v1/vms/{id}/sessions/{sid}"
          title="Delete session"
        />
        <p>Close a session.</p>
        <h3>Response</h3>
        <Fields>
          <Field name="deleted" type="Bool">
            <code>true</code> once the session is closed.
          </Field>
        </Fields>
        <h3>Example</h3>
        <CodeBlock lang="bash" code={BLOCKS[5].code} html={highlighted[5]} />
      </DocsLayout>
    </>
  );
}
