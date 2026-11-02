// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title ReputationRegistry
 * @notice Immutable Audit Subsystem for Federated Energy Theft Detection.
 * 
 * ============================================================================
 * CRITICAL SECURITY, PRIVACY & EPISTEMIC PRINCIPLES
 * ============================================================================
 * 1. ZERO RAW DATA & ZERO WEIGHTS ON-CHAIN:
 *    Under NO circumstances should raw smart meter consumption readings,
 *    consumer personally identifiable information (PII), or raw deep learning
 *    model weights be transmitted to or stored on the blockchain.
 *    Only pseudonymous client hashes (keccak256), cryptographic model update
 *    hashes (SHA-256 / Keccak-256), decisions, and reputation metrics are committed.
 *
 * 2. EPISTEMIC & OPERATIONAL BOUNDARY:
 *    The blockchain records evidence; it does NOT itself detect or prevent
 *    Byzantine poisoning. Hashes and on-chain records do NOT prove that an
 *    aggregation decision was mathematically correct—they only provide an
 *    immutable, tamper-evident, non-repudiable audit trail of what was observed,
 *    decided, and committed by the federated learning coordinator.
 * ============================================================================
 */
contract ReputationRegistry {
    // ------------------------------------------------------------------------
    // Enums & Structs
    // ------------------------------------------------------------------------

    enum Decision {
        ACCEPTED,       // 0: Valid update integrated into global model
        ATTENUATED,     // 1: Mild anomaly detected; weight downscaled
        QUARANTINED,    // 2: Byzantine adversary or extreme outlier quarantined (weight = 0)
        REJECTED        // 3: Invalid update, NaN divergence, or non-compliant format
    }

    struct AuditRecord {
        uint256 round;              // Federated learning round number
        bytes32 clientIdentifier;   // Pseudonymous client identifier (e.g. keccak256("client_A"))
        bytes32 modelUpdateHash;    // Cryptographic hash of parameter delta (Delta w)
        Decision decision;          // Aggregator disposition
        int256 reputationChange;    // Scaled by 10,000 (e.g. +500 = +0.05, -3000 = -0.30)
        uint256 newReputation;      // Scaled by 10,000 (e.g. 7500 = 0.75, 10000 = 1.00)
        uint256 timestamp;          // Block timestamp of record insertion
        bytes32 recordHash;         // Canonical hash of the full audit event payload
    }

    struct AuditRecordInput {
        uint256 round;
        bytes32 clientIdentifier;
        bytes32 modelUpdateHash;
        Decision decision;
        int256 reputationChange;
        uint256 newReputation;
        bytes32 recordHash;
    }

    // ------------------------------------------------------------------------
    // State Variables
    // ------------------------------------------------------------------------

    address public owner;
    mapping(address => bool) public authorizedCoordinators;

    // Ordered sequence of all recorded audit events
    AuditRecord[] public auditRecords;

    // Track latest recorded reputation for each client (scaled by 10,000)
    mapping(bytes32 => uint256) public latestReputation;
    mapping(bytes32 => bool) public clientRegistered;

    // Indexing for efficient historical queries
    mapping(uint256 => uint256[]) private roundToRecordIds;
    mapping(bytes32 => uint256[]) private clientToRecordIds;

    // ------------------------------------------------------------------------
    // Events
    // ------------------------------------------------------------------------

    event AuditEventRecorded(
        uint256 indexed recordId,
        uint256 indexed round,
        bytes32 indexed clientIdentifier,
        bytes32 modelUpdateHash,
        Decision decision,
        uint256 newReputation,
        uint256 timestamp,
        bytes32 recordHash
    );

    event ReputationUpdated(
        bytes32 indexed clientIdentifier,
        uint256 previousReputation,
        uint256 newReputation,
        uint256 timestamp
    );

    event CoordinatorAuthorized(address indexed coordinator);
    event CoordinatorRevoked(address indexed coordinator);

    // ------------------------------------------------------------------------
    // Modifiers
    // ------------------------------------------------------------------------

    modifier onlyOwner() {
        require(msg.sender == owner, "ReputationRegistry: caller is not owner");
        _;
    }

    modifier onlyAuthorized() {
        require(
            msg.sender == owner || authorizedCoordinators[msg.sender],
            "ReputationRegistry: caller is not authorized coordinator"
        );
        _;
    }

    // ------------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------------

    constructor() {
        owner = msg.sender;
        authorizedCoordinators[msg.sender] = true;
        emit CoordinatorAuthorized(msg.sender);
    }

    // ------------------------------------------------------------------------
    // Coordinator Access Management
    // ------------------------------------------------------------------------

    function authorizeCoordinator(address coordinator) external onlyOwner {
        require(coordinator != address(0), "Invalid coordinator address");
        authorizedCoordinators[coordinator] = true;
        emit CoordinatorAuthorized(coordinator);
    }

    function revokeCoordinator(address coordinator) external onlyOwner {
        require(coordinator != owner, "Cannot revoke contract owner");
        authorizedCoordinators[coordinator] = false;
        emit CoordinatorRevoked(coordinator);
    }

    // ------------------------------------------------------------------------
    // Audit Event Recording
    // ------------------------------------------------------------------------

    /**
     * @notice Record a single federated learning round audit event.
     * @param round FL server round.
     * @param clientIdentifier Pseudonymous client ID hash.
     * @param modelUpdateHash SHA-256 or Keccak-256 hash of gradient delta.
     * @param decision Aggregator disposition (ACCEPTED, ATTENUATED, QUARANTINED, REJECTED).
     * @param reputationChange Scaled reputation delta (+/- scaled by 10,000).
     * @param newReputation Scaled updated reputation (0..10,000).
     * @param recordHash Canonical hash of full local event payload.
     * @return recordId Unique index of the newly committed audit record.
     */
    function recordAuditEvent(
        uint256 round,
        bytes32 clientIdentifier,
        bytes32 modelUpdateHash,
        Decision decision,
        int256 reputationChange,
        uint256 newReputation,
        bytes32 recordHash
    ) external onlyAuthorized returns (uint256 recordId) {
        require(clientIdentifier != bytes32(0), "Invalid client identifier");
        require(modelUpdateHash != bytes32(0), "Invalid model update hash");
        require(newReputation <= 10000, "Reputation exceeds max scale (10000)");

        recordId = auditRecords.length;
        uint256 currentTimestamp = block.timestamp;

        AuditRecord memory record = AuditRecord({
            round: round,
            clientIdentifier: clientIdentifier,
            modelUpdateHash: modelUpdateHash,
            decision: decision,
            reputationChange: reputationChange,
            newReputation: newReputation,
            timestamp: currentTimestamp,
            recordHash: recordHash
        });

        auditRecords.push(record);
        roundToRecordIds[round].push(recordId);
        clientToRecordIds[clientIdentifier].push(recordId);

        uint256 prevRep = latestReputation[clientIdentifier];
        latestReputation[clientIdentifier] = newReputation;
        clientRegistered[clientIdentifier] = true;

        emit AuditEventRecorded(
            recordId,
            round,
            clientIdentifier,
            modelUpdateHash,
            decision,
            newReputation,
            currentTimestamp,
            recordHash
        );

        emit ReputationUpdated(
            clientIdentifier,
            prevRep,
            newReputation,
            currentTimestamp
        );

        return recordId;
    }

    /**
     * @notice Batch record multiple audit events in a single transaction to reduce gas costs.
     */
    function batchRecordAuditEvents(
        AuditRecordInput[] calldata inputs
    ) external onlyAuthorized returns (uint256[] memory recordIds) {
        recordIds = new uint256[](inputs.length);
        uint256 currentTimestamp = block.timestamp;

        for (uint256 i = 0; i < inputs.length; i++) {
            AuditRecordInput calldata inp = inputs[i];
            require(inp.clientIdentifier != bytes32(0), "Invalid client identifier");
            require(inp.modelUpdateHash != bytes32(0), "Invalid model update hash");
            require(inp.newReputation <= 10000, "Reputation exceeds max scale");

            uint256 recordId = auditRecords.length;
            recordIds[i] = recordId;

            AuditRecord memory record = AuditRecord({
                round: inp.round,
                clientIdentifier: inp.clientIdentifier,
                modelUpdateHash: inp.modelUpdateHash,
                decision: inp.decision,
                reputationChange: inp.reputationChange,
                newReputation: inp.newReputation,
                timestamp: currentTimestamp,
                recordHash: inp.recordHash
            });

            auditRecords.push(record);
            roundToRecordIds[inp.round].push(recordId);
            clientToRecordIds[inp.clientIdentifier].push(recordId);

            uint256 prevRep = latestReputation[inp.clientIdentifier];
            latestReputation[inp.clientIdentifier] = inp.newReputation;
            clientRegistered[inp.clientIdentifier] = true;

            emit AuditEventRecorded(
                recordId,
                inp.round,
                inp.clientIdentifier,
                inp.modelUpdateHash,
                inp.decision,
                inp.newReputation,
                currentTimestamp,
                inp.recordHash
            );

            emit ReputationUpdated(
                inp.clientIdentifier,
                prevRep,
                inp.newReputation,
                currentTimestamp
            );
        }

        return recordIds;
    }

    // ------------------------------------------------------------------------
    // Read & Verification Queries
    // ------------------------------------------------------------------------

    function getRecordCount() external view returns (uint256) {
        return auditRecords.length;
    }

    function getRecord(uint256 recordId) external view returns (AuditRecord memory) {
        require(recordId < auditRecords.length, "Record ID does not exist");
        return auditRecords[recordId];
    }

    function getRecordsByRound(uint256 round) external view returns (uint256[] memory) {
        return roundToRecordIds[round];
    }

    function getRecordsByClient(bytes32 clientIdentifier) external view returns (uint256[] memory) {
        return clientToRecordIds[clientIdentifier];
    }

    function getLatestReputation(bytes32 clientIdentifier) external view returns (uint256) {
        if (!clientRegistered[clientIdentifier]) {
            return 7000; // Default prior 0.70 scaled by 10,000
        }
        return latestReputation[clientIdentifier];
    }

    /**
     * @notice Verify the tamper-evident integrity of an audit record.
     * @param recordId Unique ID of the record.
     * @param expectedHash Expected canonical hash to verify against stored recordHash.
     */
    function verifyRecordIntegrity(
        uint256 recordId,
        bytes32 expectedHash
    ) external view returns (bool) {
        require(recordId < auditRecords.length, "Record ID does not exist");
        return auditRecords[recordId].recordHash == expectedHash;
    }
}
