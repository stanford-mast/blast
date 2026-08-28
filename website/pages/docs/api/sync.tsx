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
// src/api/handlers.rs::handle_sync_write on the arker-blast host. Unlike
// arker.ai's chunked/presigned sync, BLAST's write handler always does a
// single whole-file write per entry (no byte-range accumulation) and
// explicitly rejects `presigned: true`, documented as such rather than
// implying the fuller Arker behavior.
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
            The file&apos;s bytes. BLAST tries to base64-decode this first;
            if it isn&apos;t valid base64, it writes the literal text
            instead.
          </Field>
        </Fields>
        <p>
          Each write replaces the destination file in full.{" "}
          <code>start</code>/<code>end</code> byte ranges and presigned
          uploads (<code>presigned: true</code>) are accepted in the request
          shape for forward compatibility but are not implemented by this
          worker today; a presigned write returns an error result for that
          entry.
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
          <Field name="content, encoding" type='String, "utf-8" | "base64"'>
            The file&apos;s bytes and how they&apos;re encoded. Content that
            isn&apos;t valid UTF-8 comes back base64-encoded.
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
          <Field name="error" type="String">
            Set instead of the above when this entry failed (e.g. missing{" "}
            <code>content</code>, or <code>presigned: true</code>).
          </Field>
        </Fields>

        <h2 id="example">Example</h2>
        <CodeBlock lang="bash" code={BLOCKS[1].code} html={highlighted[1]} />
      </DocsLayout>
    </>
  );
}
