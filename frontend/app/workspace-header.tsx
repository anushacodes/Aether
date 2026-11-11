import Link from "next/link";

export function WorkspaceHeader({
  current,
}: {
  current: "ask" | "compare" | "admin";
}) {
  return (
    <header className="masthead">
      <Link href="/" className="wordmark">
        S3 Knowledge Engine
      </Link>
      <nav className="workspace-nav" aria-label="Workspace">
        <Link href="/" aria-current={current === "ask" ? "page" : undefined}>
          Ask a question
        </Link>
        <Link
          href="/compare"
          aria-current={current === "compare" ? "page" : undefined}
        >
          Compare sources
        </Link>
        <Link
          href="/admin"
          aria-current={current === "admin" ? "page" : undefined}
        >
          Manage corpus
        </Link>
      </nav>
    </header>
  );
}
