"use client";

import { Pencil } from "lucide-react";
import { useState, useSyncExternalStore } from "react";

const NAME_KEY = "aegistrace-profile-name";
const ROLE_KEY = "aegistrace-profile-role";
const DEFAULT_NAME = "Researcher";
const DEFAULT_ROLE = "Local session";

const listeners = new Set<() => void>();

function subscribe(callback: () => void) {
  listeners.add(callback);
  return () => listeners.delete(callback);
}

function getName(): string {
  return window.localStorage.getItem(NAME_KEY) ?? DEFAULT_NAME;
}
function getRole(): string {
  return window.localStorage.getItem(ROLE_KEY) ?? DEFAULT_ROLE;
}
function getServerName(): string {
  return DEFAULT_NAME;
}
function getServerRole(): string {
  return DEFAULT_ROLE;
}

function saveProfile(name: string, role: string) {
  window.localStorage.setItem(NAME_KEY, name);
  window.localStorage.setItem(ROLE_KEY, role);
  listeners.forEach((listener) => listener());
}

function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  return (parts[0][0] + (parts[1]?.[0] ?? "")).toUpperCase();
}

// Purely decorative, local-only identity shown at the bottom of the
// sidebar — no auth, no backend user table, nothing sent anywhere. Stored
// per-browser in localStorage and editable in place. Deliberately not tied
// to any account system, per this project's explicit no-authentication
// scope.
export function ProfileBlock() {
  const name = useSyncExternalStore(subscribe, getName, getServerName);
  const role = useSyncExternalStore(subscribe, getRole, getServerRole);
  const [editing, setEditing] = useState(false);
  const [draftName, setDraftName] = useState("");
  const [draftRole, setDraftRole] = useState("");

  if (editing) {
    return (
      <form
        className="space-y-2 border-t border-border px-4 py-3.5"
        onSubmit={(e) => {
          e.preventDefault();
          saveProfile(draftName.trim() || DEFAULT_NAME, draftRole.trim() || DEFAULT_ROLE);
          setEditing(false);
        }}
      >
        <input
          className="input text-[13px]"
          placeholder="Your name"
          defaultValue={name}
          autoFocus
          onChange={(e) => setDraftName(e.target.value)}
        />
        <input
          className="input text-[13px]"
          placeholder="Role"
          defaultValue={role}
          onChange={(e) => setDraftRole(e.target.value)}
        />
        <div className="flex gap-2">
          <button type="submit" className="flex-1 rounded-md bg-primary px-2 py-1 text-[12px] font-medium text-primary-foreground">
            Save
          </button>
          <button
            type="button"
            onClick={() => setEditing(false)}
            className="flex-1 rounded-md border border-border px-2 py-1 text-[12px] text-muted-foreground"
          >
            Cancel
          </button>
        </div>
      </form>
    );
  }

  return (
    <button
      type="button"
      onClick={() => {
        setDraftName(name);
        setDraftRole(role);
        setEditing(true);
      }}
      className="group flex items-center gap-2.5 border-t border-border px-5 py-3.5 text-left transition-colors hover:bg-muted"
    >
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary-tint text-[12px] font-semibold text-primary">
        {initials(name)}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[13px] font-medium text-foreground">{name}</span>
        <span className="block truncate text-[11px] text-muted-foreground">{role}</span>
      </span>
      <Pencil className="h-3.5 w-3.5 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
    </button>
  );
}
