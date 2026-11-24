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
  {
    lang: "bash",
    code: `sudo mkdir -p /opt/smolvm
curl -fsSL "https://github.com/smol-machines/smolvm/releases/download/v1.37.0/smolvm-1.37.0-linux-$(uname -m | sed s/aarch64/arm64/).tar.gz" \\
  | sudo tar -xz -C /opt/smolvm --strip-components=1
sudo ln -sf /opt/smolvm/smolvm /usr/local/bin/smolvm
sudo usermod -aG kvm "$USER"   # takes effect at your next login

cargo install --locked --git https://github.com/stanford-mast/blast
blast`,
  },
  {
    lang: "bash",
    code: `cat > blast.toml <<'EOF'
[resources]
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
        <p>
          Install BLAST on the local or BYOC machine, a Linux machine with
          KVM. BLAST runs the SmolVM 1.37.0 release, and building it needs a
          Rust toolchain and a C compiler:
        </p>
        <CodeBlock lang="bash" code={BLOCKS[0].code} html={highlighted[0]} />
        <p>
          SmolVM runs unprivileged, so BLAST&apos;s user must be able to open{" "}
          <code>/dev/kvm</code>; on a host without a <code>kvm</code> group, run{" "}
          <code>sudo groupadd -r kvm &amp;&amp; sudo chgrp kvm /dev/kvm &amp;&amp; sudo chmod 660 /dev/kvm</code>{" "}
          before <code>usermod</code>.
        </p>

        <h2 id="configure-a-pool">Configure a pool</h2>
        <p>Configure a pool of CPU, memory, and disk:</p>
        <CodeBlock lang="bash" code={BLOCKS[1].code} html={highlighted[1]} />
        <p>
          Set <code>[api] token</code>, or keep the worker on localhost:
          without a token, a standalone worker&apos;s API is open to anyone
          who can reach it.
        </p>

        <h2 id="fork-and-run">Fork and run</h2>
        <p>Invoke the BLAST API to fork and run sandboxes:</p>
        <CodeBlock lang="bash" code={BLOCKS[2].code} html={highlighted[2]} />
      </DocsLayout>
    </>
  );
}
