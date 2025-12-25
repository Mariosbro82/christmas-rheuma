//
//  CoreDataErrorHandler.swift
//  InflamAI-Swift
//
//  Created by Trae AI on 2024.
//

import Foundation
import CoreData
import SwiftUI

// MARK: - Core Data Error Types
enum CoreDataError: LocalizedError {
    case saveFailure(Error)
    case fetchFailure(Error)
    case deleteFailure(Error)
    case entityNotFound(String)
    case invalidData(String)
    case contextUnavailable
    
    var errorDescription: String? {
        switch self {
        case .saveFailure(let error):
            return "Failed to save data: \(error.localizedDescription)"
        case .fetchFailure(let error):
            return "Failed to fetch data: \(error.localizedDescription)"
        case .deleteFailure(let error):
            return "Failed to delete data: \(error.localizedDescription)"
        case .entityNotFound(let entityName):
            return "Entity '\(entityName)' not found in Core Data model"
        case .invalidData(let message):
            return "Invalid data: \(message)"
        case .contextUnavailable:
            return "Core Data context is unavailable"
        }
    }
    
    var recoverySuggestion: String? {
        switch self {
        case .saveFailure:
            return "Please try again. If the problem persists, restart the app."
        case .fetchFailure:
            return "Please refresh the data or restart the app."
        case .deleteFailure:
            return "Please try again. The item may have already been deleted."
        case .entityNotFound:
            return "Please update the app to the latest version."
        case .invalidData:
            return "Please check your input and try again."
        case .contextUnavailable:
            return "Please restart the app."
        }
    }
}

// MARK: - Core Data Error Handler
class CoreDataErrorHandler: ObservableObject {
    @Published var currentError: CoreDataError?
    @Published var showingError = false
    
    static let shared = CoreDataErrorHandler()
    
    private init() {}
    
    func handle(_ error: CoreDataError) {
        DispatchQueue.main.async {
            self.currentError = error
            self.showingError = true
        }
        
        // Log error for debugging
        print("CoreData Error: \(error.localizedDescription)")
        if let suggestion = error.recoverySuggestion {
            print("Recovery Suggestion: \(suggestion)")
        }
    }
    
    func clearError() {
        currentError = nil
        showingError = false
    }
}

// MARK: - Core Data Operations Helper
struct CoreDataOperations {
    
    // MARK: - Safe Save Operation
    static func safeSave(context: NSManagedObjectContext, completion: @escaping (Result<Void, CoreDataError>) -> Void) {
        guard context.hasChanges else {
            completion(.success(()))
            return
        }
        
        do {
            try context.save()
            completion(.success(()))
        } catch {
            let coreDataError = CoreDataError.saveFailure(error)
            CoreDataErrorHandler.shared.handle(coreDataError)
            completion(.failure(coreDataError))
        }
    }
    
    // MARK: - Safe Fetch Operation
    static func safeFetch<T: NSManagedObject>(
        request: NSFetchRequest<T>,
        context: NSManagedObjectContext,
        completion: @escaping (Result<[T], CoreDataError>) -> Void
    ) {
        do {
            let results = try context.fetch(request)
            completion(.success(results))
        } catch {
            let coreDataError = CoreDataError.fetchFailure(error)
            CoreDataErrorHandler.shared.handle(coreDataError)
            completion(.failure(coreDataError))
        }
    }
    
    // MARK: - Safe Delete Operation
    static func safeDelete(
        object: NSManagedObject,
        context: NSManagedObjectContext,
        completion: @escaping (Result<Void, CoreDataError>) -> Void
    ) {
        context.delete(object)
        safeSave(context: context, completion: completion)
    }
    
    // MARK: - Create Entity Safely
    static func createEntity<T: NSManagedObject>(
        entityName: String,
        context: NSManagedObjectContext
    ) -> Result<T, CoreDataError> {
        guard let entity = NSEntityDescription.entity(forEntityName: entityName, in: context) else {
            let error = CoreDataError.entityNotFound(entityName)
            CoreDataErrorHandler.shared.handle(error)
            return .failure(error)
        }
        
        let object = T(entity: entity, insertInto: context)
        return .success(object)
    }
    
    // MARK: - Batch Delete Operation
    static func batchDelete<T: NSManagedObject>(
        fetchRequest: NSFetchRequest<T>,
        context: NSManagedObjectContext,
        completion: @escaping (Result<Void, CoreDataError>) -> Void
    ) {
        let deleteRequest = NSBatchDeleteRequest(fetchRequest: fetchRequest as! NSFetchRequest<NSFetchRequestResult>)
        deleteRequest.resultType = .resultTypeObjectIDs
        
        do {
            let result = try context.execute(deleteRequest) as? NSBatchDeleteResult
            let objectIDArray = result?.result as? [NSManagedObjectID]
            let changes = [NSDeletedObjectsKey: objectIDArray ?? []]
            NSManagedObjectContext.mergeChanges(fromRemoteContextSave: changes, into: [context])
            completion(.success(()))
        } catch {
            let coreDataError = CoreDataError.deleteFailure(error)
            CoreDataErrorHandler.shared.handle(coreDataError)
            completion(.failure(coreDataError))
        }
    }
}

// MARK: - SwiftUI Error Alert Modifier
struct CoreDataErrorAlert: ViewModifier {
    @StateObject private var errorHandler = CoreDataErrorHandler.shared
    
    func body(content: Content) -> some View {
        content
            .alert("Error", isPresented: $errorHandler.showingError) {
                Button("OK") {
                    errorHandler.clearError()
                }
            } message: {
                if let error = errorHandler.currentError {
                    VStack(alignment: .leading) {
                        Text(error.localizedDescription)
                        if let suggestion = error.recoverySuggestion {
                            Text(suggestion)
                                .font(.caption)
                        }
                    }
                }
            }
    }
}

extension View {
    func coreDataErrorAlert() -> some View {
        modifier(CoreDataErrorAlert())
    }
}