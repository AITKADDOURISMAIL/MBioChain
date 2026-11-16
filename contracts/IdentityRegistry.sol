// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title MBioChain identity registry (Ethereum-compatible version of mbiochain/ledger.py)
/// @notice Stores, per DID, the address derived from the biometric secp256k1 key, the hash of the
///         off-chain helper data, the key epoch and a guardian set for social recovery.
///         No biometric data is ever stored on-chain.
contract IdentityRegistry {
    struct Identity {
        address key;            // address of the current biometric-derived key
        bytes32 helperHash;     // SHA-256 of the off-chain helper data (IPFS / device)
        uint32 epoch;           // incremented at each rotation / recovery
        uint8 threshold;        // guardians needed for recovery
        address[] guardians;
        bool exists;
    }

    struct RecoveryRequest {
        address newKey;
        bytes32 newHelperHash;
        uint32 newEpoch;
        uint8 approvals;
    }

    mapping(bytes32 => Identity) private ids;                 // did hash -> identity
    mapping(bytes32 => RecoveryRequest) public pending;       // did hash -> pending recovery
    mapping(bytes32 => uint256) public requestNonce;                                   // did hash -> request id
    mapping(bytes32 => mapping(uint256 => mapping(address => bool))) private voted;
    mapping(address => bool) public revoked;

    event Registered(bytes32 indexed did, address key, bytes32 helperHash);
    event Rotated(bytes32 indexed did, address oldKey, address newKey, uint32 epoch);
    event RecoveryApproved(bytes32 indexed did, address guardian, uint8 approvals);
    event Recovered(bytes32 indexed did, address oldKey, address newKey, uint32 epoch);

    function register(bytes32 did, bytes32 helperHash, address[] calldata guardians, uint8 threshold) external {
        require(!ids[did].exists, "DID exists");
        require(!revoked[msg.sender], "key revoked");
        require(threshold > 0 && threshold <= guardians.length, "bad threshold");
        Identity storage id = ids[did];
        id.key = msg.sender;
        id.helperHash = helperHash;
        id.threshold = threshold;
        id.guardians = guardians;
        id.exists = true;
        emit Registered(did, msg.sender, helperHash);
    }

    /// Voluntary rotation: signed (sent) by the current biometric key.
    function rotate(bytes32 did, address newKey, bytes32 newHelperHash) external {
        Identity storage id = ids[did];
        require(id.exists && msg.sender == id.key, "not current key");
        require(!revoked[newKey], "new key revoked");
        _switch(did, id, newKey, newHelperHash, id.epoch + 1);
        emit Rotated(did, msg.sender, newKey, id.epoch);
    }

    /// Social recovery: each guardian approves the same (newKey, newHelperHash).
    function approveRecovery(bytes32 did, address newKey, bytes32 newHelperHash) external {
        Identity storage id = ids[did];
        require(id.exists && _isGuardian(id, msg.sender), "not guardian");
        uint32 target = id.epoch + 1;
        RecoveryRequest storage r = pending[did];
        if (r.newEpoch != target || r.newKey != newKey || r.newHelperHash != newHelperHash) {
            requestNonce[did] += 1;                                             // new request resets votes
            pending[did] = RecoveryRequest(newKey, newHelperHash, target, 0);
            r = pending[did];
        }
        uint256 rid = requestNonce[did];
        require(!voted[did][rid][msg.sender], "already approved");
        voted[did][rid][msg.sender] = true;
        r.approvals += 1;
        emit RecoveryApproved(did, msg.sender, r.approvals);
        if (r.approvals >= id.threshold) {
            address old = id.key;
            _switch(did, id, newKey, newHelperHash, target);
            delete pending[did];
            requestNonce[did] += 1;
            emit Recovered(did, old, newKey, target);
        }
    }

    function isCurrentKey(bytes32 did, address key) external view returns (bool) {
        return ids[did].exists && ids[did].key == key && !revoked[key];
    }

    function identity(bytes32 did) external view returns (address, bytes32, uint32, uint8, address[] memory) {
        Identity storage id = ids[did];
        return (id.key, id.helperHash, id.epoch, id.threshold, id.guardians);
    }

    function _switch(bytes32, Identity storage id, address newKey, bytes32 h, uint32 epoch) private {
        revoked[id.key] = true;
        id.key = newKey;
        id.helperHash = h;
        id.epoch = epoch;
    }

    function _isGuardian(Identity storage id, address a) private view returns (bool) {
        for (uint256 i = 0; i < id.guardians.length; i++) if (id.guardians[i] == a) return true;
        return false;
    }
}
