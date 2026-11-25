/** Preload used by frontend tests to block every network API before connecting. */

import dns from "node:dns";
import http from "node:http";
import https from "node:https";
import net from "node:net";
import tls from "node:tls";

const blocked = (api) => (..._args) => {
  throw new Error(`network disabled in frontend tests: ${api}`);
};

for (const name of [
  "lookup",
  "lookupService",
  "resolve",
  "resolve4",
  "resolve6",
  "resolveAny",
  "resolveCaa",
  "resolveCname",
  "resolveMx",
  "resolveNaptr",
  "resolveNs",
  "resolvePtr",
  "resolveSoa",
  "resolveSrv",
  "resolveTxt",
  "reverse",
]) {
  if (typeof dns[name] === "function") dns[name] = blocked(`dns.${name}`);
}

net.connect = blocked("net.connect");
net.createConnection = blocked("net.createConnection");
tls.connect = blocked("tls.connect");
http.request = blocked("http.request");
http.get = blocked("http.get");
https.request = blocked("https.request");
https.get = blocked("https.get");
globalThis.fetch = blocked("fetch");
