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
  {
    lang: "bash",
    code: `curl -X POST localhost:7240/v1/fork \\
  -H "Content-Type: application/json" \\
  -d '{"image":"ubuntu:24.04"}'`,
  },
  {
    lang: "bash",
    code: `curl -X POST localhost:7240/v1/fork \\
  -H "Content-Type: application/json" \\
  -d "{\\"source_vm_id\\":\\"$VM\\",\\"name\\":\\"feature-xyz\\"}"`,
  },
  {
    lang: "bash",
    code: `curl -X POST localhost:7240/v1/fork \\
  -H "Content-Type: application/json" \\
  -d '{
    "image": "ghcr.io/org/private-image:latest",
    "registry_auth": { "username": "myuser", "password": "$GHCR_TOKEN" }
  }'`,
  },
];

export async function getStaticProps() {
  return { props: { highlighted: await highlightBlocks(BLOCKS) } };
}

// Cross-referenced against src/api/types.rs::ForkRequest and
// src/api/handlers.rs::handle_fork on the arker-blast host. `network` is part
// of the JSON shape but unused by handle_fork today, so it's left undocumented
// rather than advertising a field that silently has no effect.
export default function ApiForkPage({ highlighted }: { highlighted: string[] }) {
  return (
    <>
      <Head>
        <title>Fork | BLAST Documentation</title>
      </Head>
      <DocsLayout toc={toc}>
        <Endpoint method="POST" path="/v1/fork" />
        <p>Create a VM, either from an OCI image or by forking an existing VM.</p>

        <h2 id="request">Request</h2>
        <p>
          Specify exactly one of <code>image</code>, <code>source_vm_id</code>
          , or <code>source_vm_name</code>. If <code>image</code> is present
          it takes precedence over the other two.
        </p>
        <Fields>
          <Field name="image" type="String" required="Required, one of">
            OCI image reference, e.g. <code>ubuntu:24.04</code>. Boots a
            rootfs converted from the image.
          </Field>
          <Field
            name="source_vm_id"
            type="String"
            required="Required, one of"
          >
            Id of an existing VM to fork from.
          </Field>
          <Field
            name="source_vm_name"
            type="String"
            required="Required, one of"
          >
            Name of an existing VM to fork from.
          </Field>
          <Field name="name" type="String">
            Name for the new VM.
          </Field>
          <Field name="resources" type="Object">
            <code>
              {String.raw`{ "vcpu": 2, "memory_mib": 2048, "disk_mib": 10240 }`}
            </code>{" "}
            is the default; any field you omit falls back to it.
          </Field>
          <Field name="registry_auth" type="Object">
            Credentials for pulling <code>image</code> from a private
            registry (ghcr.io, a private Docker Hub repo, ECR, etc):{" "}
            <code>
              {String.raw`{ "username": "...", "password": "..." }`}
            </code>
            . Used only for this one pull and never stored. For AWS ECR use
            username <code>AWS</code> with the output of{" "}
            <code>aws ecr get-login-password</code>; for Docker Hub or GHCR
            use your username and a personal access token as the password.
            Ignored when forking from a source VM, which has no registry to
            authenticate against.
          </Field>
        </Fields>

        <h2 id="response">Response</h2>
        <p>
          Returns a{" "}
          <a href="/docs/api/vms#the-vm-object" className="link">
            VM object
          </a>{" "}
          for the new machine.
        </p>

        <h2 id="example">Example</h2>
        <p>Fork from an image:</p>
        <CodeBlock lang="bash" code={BLOCKS[0].code} html={highlighted[0]} />

        <p>Fork from a running VM:</p>
        <CodeBlock lang="bash" code={BLOCKS[1].code} html={highlighted[1]} />

        <p>Fork from a private image:</p>
        <CodeBlock lang="bash" code={BLOCKS[2].code} html={highlighted[2]} />
      </DocsLayout>
    </>
  );
}
