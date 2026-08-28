import Head from "next/head";
import { DocsLayout, type TocItem } from "@/components/docs/DocsLayout";
import { CodeBlock } from "@/components/docs/CodeBlock";
import { highlightBlocks } from "@/lib/docs/highlight";

const toc: TocItem[] = [
  { id: "install", text: "Install" },
  { id: "configure-a-pool", text: "Configure a pool" },
  { id: "fork-and-run", text: "Fork and run" },
];

const BLOCKS = [
  { lang: "bash", code: `cargo install blast_core\nblast` },
  {
    lang: "bash",
    code: `cat > blast.toml <<'EOF'
[worker.resources]
vcpu = 8
memory_mib = 16384
disk_mib = 102400
EOF

blast --config blast.toml`,
  },
  {
    lang: "bash",
    code: `VM1=$(curl -s -X POST localhost:7240/v1/fork \\
  -H "Content-Type: application/json" \\
  -d '{"image":"ubuntu:24.04"}' | jq -r .vm_id)

VM2=$(curl -s -X POST localhost:7240/v1/fork \\
  -H "Content-Type: application/json" \\
  -d "{\\"source_vm_id\\":\\"$VM1\\",\\"name\\":\\"feature-xyz\\"}" | jq -r .vm_id)

curl -X POST localhost:7240/v1/vms/$VM2/runs \\
  -H "Content-Type: application/json" \\
  -d '{"command":"echo hello from fork"}'`,
  },
];

export async function getStaticProps() {
  return { props: { highlighted: await highlightBlocks(BLOCKS) } };
}

export default function QuickstartPage({ highlighted }: { highlighted: string[] }) {
  return (
    <>
      <Head>
        <title>Quickstart | BLAST Documentation</title>
        <meta
          name="description"
          content="Install BLAST, configure a resource pool, and fork/run sandboxes."
        />
      </Head>
      <DocsLayout toc={toc}>
        <h1>Quickstart</h1>

        <h2 id="install">Install</h2>
        <p>Install BLAST on the local or BYOC machine:</p>
        <CodeBlock lang="bash" code={BLOCKS[0].code} html={highlighted[0]} />

        <h2 id="configure-a-pool">Configure a pool</h2>
        <p>Configure a pool of CPU, memory, and disk:</p>
        <CodeBlock lang="bash" code={BLOCKS[1].code} html={highlighted[1]} />

        <h2 id="fork-and-run">Fork and run</h2>
        <p>Invoke the BLAST API to fork and run sandboxes:</p>
        <CodeBlock lang="bash" code={BLOCKS[2].code} html={highlighted[2]} />
      </DocsLayout>
    </>
  );
}
