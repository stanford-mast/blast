import Head from "next/head";
import { DocsLayout, type TocItem } from "@/components/docs/DocsLayout";

const toc: TocItem[] = [
  { id: "purpose", text: "Purpose" },
  { id: "features", text: "Features" },
];

// Body is the "Purpose" section of the repo README, reused verbatim (not
// rephrased) per the brief, with one deliberate exception: the trailing
// "such as [Arker](https://arker.ai)" example is dropped from the
// VM-providers bullet in favor of generic "a control plane" wording, same
// as the Orchestration page.
export default function IntroductionPage() {
  return (
    <>
      <Head>
        <title>Introduction | BLAST Documentation</title>
        <meta
          name="description"
          content="BLAST is a single binary for local sandbox orchestration given a pool of CPU, memory, disk."
        />
      </Head>
      <DocsLayout toc={toc}>
        <h1>Introduction</h1>
        <h2 id="purpose">Purpose</h2>
        <p>
          BLAST is a single binary for local sandbox orchestration given a
          pool of CPU, memory, disk. More precisely, BLAST abstracts over
          local sandboxes such as SmolVM, Hypeman, Docker to provides a
          simple API to fork and run sandboxed commands, sync data, and
          monitor VMs, sessions, runs while automatically scheduling and
          placing forks and runs, snapshots, syncing snapshots to durable
          storage, migration, and managing resource pressure.
        </p>
        <h2 id="features">Features</h2>
        <p>
          Compared to existing solutions for local sandboxing or BYOC
          sandbox compute, BLAST features:
        </p>
        <ul>
          <li>
            <strong>Single 7 MB binary.</strong> No Terraform. No Packer. No
            extra dependencies.
          </li>
          <li>
            <strong>Most permissible license.</strong> Code is MIT-licensed
            and just 3,586 lines, built to keep enterprise security reviews
            as simple as possible.
          </li>
          <li>
            <strong>Full orchestration.</strong> Unlike other solutions that
            simply provide utilities for creating sandboxes, BLAST takes a
            given pool of CPU, memory, disk and optimally serves forking and
            running sandboxed commands.
          </li>
          <li>
            <strong>Compatible with VM providers.</strong> For a unified
            control plane across bursty cloud compute and user-provided
            compute, BLAST is built to integrate with many VM providers.
          </li>
          <li>
            Built and actively maintained by a growing team of open-source
            sandboxing and orchestration enthusiasts
          </li>
        </ul>
      </DocsLayout>
    </>
  );
}
