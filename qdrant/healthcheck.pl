#!/usr/bin/env perl
use IO::Socket::INET;
my $key = $ENV{QDRANT__SERVICE__API_KEY} // "";
my $sock = IO::Socket::INET->new(PeerAddr => "127.0.0.1", PeerPort => 6333, Timeout => 3) or exit 1;
print $sock "GET / HTTP/1.1\r\nHost: localhost\r\napi-key: $key\r\nConnection: close\r\n\r\n";
my $resp = <$sock>;
close $sock;
#warn "Healthcheck response: $resp\n";
exit($resp && $resp =~ /HTTP\/\d\.\d\s+200/ ? 0 : 1);
