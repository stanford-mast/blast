import Head from "next/head";
import { DocsLayout, type TocItem } from "@/components/docs/DocsLayout";
import { Endpoint } from "@/components/docs/Endpoint";
import { Fields, Field } from "@/components/docs/Fields";
import { CodeBlock } from "@/components/docs/CodeBlock";
import { highlightBlocks } from "@/lib/docs/highlight";

const toc: TocItem[] = [
  { id: "request", text: "Request" },
  { id: "response", text: "Response" },
  { id: "example", text: "Example" },
];

const BLOCKS = [
  { lang: "json", code: `{ "op": "read", "path": "/absolute/or/relative/path" }` },
  {
    lang: "bash",
    code: `curl -X POST localhost:7240/v1/vms/$VM/sync \\
  -H "Content-Type: application/json" \\
  -d '{"op":"read","path":"/tmp/hello.txt"}'`,
  },
];

export async function getStaticProps() {
  return { props: { highlighted: await highlightBlocks(BLOCKS) } };
}

// Cross-referenced against src/api/types.rs (SyncRequest/SyncResponse) and
// src/api/sync.rs, which refuses `presigned: true`.
export default function ApiSyncPage({ highlighted }: { highlighted: string[] }) {
  return (
    <>
      <Head>
        <title>Sync | BLAST Documentation</title>
      </Head>
      <DocsLayout toc={toc}>
        <Endpoint method="POST" path="/v1/vms/{id}/sync" />
        <p>Read or write a file inside a VM.</p>

        <h2 id="request">Request</h2>
        <p>
          The body is discriminated by <code>op</code>.
        </p>
        <h3>Read</h3>
        <CodeBlock lang="json" code={BLOCKS[0].code} html={highlighted[0]} />
        <h3>Write</h3>
        <p>
          <code>{String.raw`{ "op": "write", "writes": [...] }`}</code>, where
          each entry is:
        </p>
        <Fields>
          <Field name="path" type="String" required>
            Destination path inside the VM.
          </Field>
          <Field name="content" type="String" required>
            The file&apos;s bytes, base64-encoded.
          </Field>
        </Fields>
        <p>
          Each write replaces the destination file in full. A large file can
          be sent in chunks of up to 5 MiB, each entry giving the file&apos;s{" "}
          <code>size</code>, a shared <code>upload_id</code>, and its{" "}
          <code>start</code>/<code>end</code> byte range; the file is written
          once every byte has arrived. Presigned uploads (
          <code>presigned: true</code>) are not implemented by this worker
          today; a request with one is refused with{" "}
          <code>unsupported_operation</code>.
        </p>

        <h2 id="response">Response</h2>
        <p>A read returns:</p>
        <Fields>
          <Field name="ok" type="Bool">
            <code>true</code> on success.
          </Field>
          <Field name="path" type="String">
            Echoes the request path.
          </Field>
          <Field name="size" type="Int">
            Byte length of <code>content</code>.
          </Field>
          <Field name="content, encoding" type='String, "utf8" | "base64"'>
            The file&apos;s bytes and how they&apos;re encoded. Content that
            isn&apos;t valid UTF-8 comes back base64-encoded. Reads are
            limited to 16 MiB.
          </Field>
        </Fields>
        <p>
          A write returns{" "}
          <code>{String.raw`{ "ok": ..., "op": "write", "results": [...] }`}</code>
          , one result per request entry:
        </p>
        <Fields>
          <Field name="received_bytes" type="Int">
            Bytes decoded from <code>content</code>.
          </Field>
          <Field name="complete, written" type="Bool">
            <code>true</code> once the file is written.
          </Field>
          <Field name="error" type="Object">
            <code>{String.raw`{ "code", "message" }`}</code>, set instead of
            the above when this entry failed.
          </Field>
        </Fields>

        <h2 id="example">Example</h2>
        <CodeBlock lang="bash" code={BLOCKS[1].code} html={highlighted[1]} />
      </DocsLayout>
    </>
  );
}
